import _pickle
import random
import socket
import sys
import threading
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

HOST = "127.0.0.1"


# -------------------------------------------------------------------------------------
# -------------------------------------- METHODS --------------------------------------
# -------------------------------------------------------------------------------------

def receive_model_from_client(server, _socket):
    """
    This method will be run by a thread started by the server. It's job is to wait for client's updated model after the global one has been sent.
    We want this to be made by a thread because we want the server to be able to send the global model to all clients at the same time (so without
    waiting for them to respond) and effectively perform a parallele training.
    :param server: server starting the thread. The thread will modify some attributes of the server
    :param _socket: i-th socket used for sending the global model to i-th client
    """
    try:
        received = receive_all(_socket)
        try:
            received_packet = _pickle.loads(received)

            # UNPACK THE PACKET
            client_id = received_packet["id"]
            clients_model = received_packet["model"]
            local_training_loss = received_packet['local_training_loss']
            global_model_accuracy = received_packet['global_model_accuracy']

            server.loss.append(local_training_loss) # add the loss to the server's loss history
            server.accuracy.append(global_model_accuracy)  # add the accuracy to the server's accuracy history

            server.round_losses.append(local_training_loss)  # add the loss the round losses
            server.round_accuracies.append(global_model_accuracy) # add the accuracy to the rounf accuracies

            server.clients_models[client_id] = clients_model  # add the received model to the dictionary of received models at this round


        except Exception as e:
            print(f"Exception is: {e}")
            pass
        _socket.close()
    except socket.error as e:
        pass


def receive_all(sock):
    """
    Due to high dimensional packets (about 60k bytes) exchanged between server and client, we need to make the read of the received message incremental,
    with steps equal to the maximum supported size: 4096 bytes. The reading of the message will be buffered
    :param sock: socket to receive from
    :return: the whole data in bytes
    """
    BUFF_SIZE = 4096
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data


# -------------------------------------------------------------------------------------
# -------------------------------------- CLASSES --------------------------------------
# -------------------------------------------------------------------------------------


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)
        self.fc1.weight.data = torch.randn(self.fc1.weight.size()) * .01  # generate random initial model

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor.
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


class HandshakeThread(threading.Thread):
    # The handshake thread is owned by the server and it is used for keep listening
    # for new connections while the training has already started.
    # In this way clients can join at any moment in time and start
    # collaborating for the training of the model
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self) -> None:
        while server.alive:
            self.server.server_socket.listen()
            conn, addr = self.server.server_socket.accept()
            received = conn.recv(4096)
            client = _pickle.loads(received)
            print(f"Client connected: {client}")
            self.server.clients.append(client)  # append the client to list of alive clients


class Server:
    def __init__(self):
        self.port_no = sys.argv[1]  # fixed to 6000
        self.client_subsampling = sys.argv[2]  # 0 for use all the clients' models, 1 use only 2 chosen randomly
        if self.client_subsampling == "0":
            self.client_subsampling = False
        else:
            self.client_subsampling = True

        self.alive = False

        self.clients = list()  # list of objects { "id": int, "port_no": int, "data_size": int, "model_sent": int }
        self.clients_models = dict()  # (key = ID, value = model)

        self.model = MCLR()  # model to be trained

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, int(self.port_no)))

        self.communication_rounds = 100

        self.round_losses = []  # will store the local training losses received from clients at each round, gets emptied after it is used for calculating the average
        self.round_accuracies = []  # will store the global model accuracy among clients at each round, gets emptied after it is used for calculating the average

        self.loss = []  # will store every local training loss sent by the client during the entire execution, used for computing final overall average
        self.accuracy = []  # will store every global model accuracy sent by the client during the entire execution, used for computing final overall average

        self.global_average_accuracy = -1  # will store the global average accuracy at each round. Init at -1 for init model
        self.global_average_training_loss = -1  # will store the global average training loss at each round. Init at -1 for init model

        with open("server_average_loss_accuracy.csv", 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(["communication_round", "global_average_accuracy", "global_average_loss"])

    def run(self):
        self.alive = True

        handshake_thread = HandshakeThread(self)  # create handshake listener thread

        # LISTEN FOR FIRST HANDSHAKE
        self.server_socket.listen()  # start listening for the first client to connect
        conn, addr = self.server_socket.accept()  # accept connection of the first client
        received = conn.recv(4096)
        client = _pickle.loads(received)
        self.clients.append(client)  # append the new client to the alive clients list
        print(f"Client connected: {client}")

        # START LISTENING FOR OTHER HANDSHAKES
        handshake_thread.start()
        time.sleep(10)  # timer after which the federated learning starts
        start_time = time.time()
        # START FEDERATED LEARNING
        self.federated_learning()

        # EVALUATE GLOBAL MODEL AT THE END OF THE TRAINING
        print("finished learning: evaluating model...")
        average_loss = sum(self.loss) / len(self.loss)
        average_accuracy = sum(self.accuracy) / len(self.accuracy)
        print(f"Average Loss is: {average_loss}")
        print(f"Average Accuracy is: {average_accuracy}")

        # SAVE ALL THE LOSSES AND ACCURACIES COLLECTED DURING EXECUTION TO LOG FILE
        with open("server_log.csv", 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(["loss", "accuracy"])
            for loss, accuracy in zip(self.loss, self.accuracy):
                writer.writerow([float(loss), accuracy])

        # PRINT EXECUTION TIME
        print(f"Training time: {time.time() - start_time}")

        # DIE
        self.alive = False

    def federated_learning(self):
        """
        Runs the federated learning algorithm
        """
        for communication_round in range(self.communication_rounds):
            print(f"Global Iteration: {communication_round + 1}")
            print(f"Total number of clients: {len(self.clients)}")

            # SEND GLOBAL MODEL TO EACH CLIENT
            self.broadcast_global_model(self.communication_rounds - 1 - communication_round)

            # CHECK THAT EVERY CLIENT SENT THE UPDATED MODEL
            received_all_models = self.check_received_all_models()
            if not received_all_models:
                self.handle_dead_clients()

            # SELECT CLIENTS FOR MODEL AGGREGATION (depending on subsampling mode)
            if self.client_subsampling:
                selected_clients = self.subsample_clients()
            else:
                selected_clients = self.clients

            # IF NO CLIENTS ARE ALIVE, you cannot calculate the new global model
            if len(selected_clients) == 0:
                self.clients_models = dict()
                continue

            # GET TOTAL DATA SIZE (sum of data sizes of the alive clients)
            total_size = self.get_total_size()

            # COMPUTE NEW GLOBAL MODEL
            self.compute_new_global_model(selected_clients, total_size)

            # COMPUTE GLOBAL AVERAGES
            self.compute_global_averages()

            # LOG NEW GLOBAL AVERAGES
            self.log_global_averages(communication_round)

            # CLEAN MODELS DICTIONARY FOR NEXT ITERATION
            self.clients_models = dict()

    def handle_dead_clients(self):
        """
        If the self.check_received_all_models detected a possible failure, removes the actually failed clients from the alive clients list
        """
        for client in self.clients:
            # IF
            # the list of alive clients contains client ids that do not compare among the received models
            # AND
            # the clients corresponding to those ids have already sent a local model in the past
            # THEN remove those clients from the list of alive clients

            # the check client['model_sent'] == 1 is done because it may happen that a client joins the network while
            # the server is waiting for all the models to be received. In this case, the self.check_received_all_models()
            # will return True, because the new client is in the alive clients but has not sent a local model to the server
            # (it never even received the global model for the first time!).
            # By checking the model_sent attribute, we know if the client has already contributed to the training before (model_sent = 1)
            # and so now is actually failed, or if it has just joined (model_sent = 0) and so it doesn't have to be removed from the list of alive clients
            if not self.clients_models.keys().__contains__(client["id"]) and client['model_sent'] == 1:
                # client dead and will be removed from the clients list
                print(f"------- CLIENT {client['id']} DIED --------")
                self.clients.remove(client)

    def check_received_all_models(self) -> bool:
        """
        Checks for a maximum of 5 seconds if all the models have been received by the server
        if this happens before 5 seconds, then it means that all the alive clients sent their model
        otherwise it means that a previously alive client died and we need to handle this
        :return: True if all the alive clients sent their model, False if one or more clients died and didn't send the model
        """
        checks = 0
        all_received = False
        while checks < 10 and not all_received:
            if len(self.clients) == len(
                    list(
                        self.clients_models.keys())):  # if the number of received models is equal to the number of alive clients
                all_received = True
            else:
                checks += 1
                time.sleep(0.5)
        return all_received

    def broadcast_global_model(self, rounds_left):
        """
        Sends the global model to each alive client and for each of them starts a thread that listens for the updated model.
        :param rounds_left: rounds left for completing the federated training, will be sent to clients to let them know when to stop themselves
        """
        print("Broadcasting global model")
        for client in self.clients:
            _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # PREPARE MODEL PACKAGE

                package = {
                    "model": self.model,  # global model
                    "rounds_left": rounds_left,  # let the clients know when to stop
                    "global_average_accuracy": self.global_average_accuracy,  # send to clients so that they print it at terminal
                    "global_average_training_loss": self.global_average_training_loss   # send to clients so that they print it at terminal
                }
                message = _pickle.dumps(package)

                # SEND MODEL
                _socket.connect((HOST, int(client["port_no"])))
                _socket.sendall(message)
                client['model_sent'] = 1  # sent model_sent to 1. Useful for understanding and handling client failure.
                # model_Sent = 1 means that the client received already the global model (at least) for the first time
                # and so if it doesn't send back the model, it is considered to be failed.

                # START NEW THREAD THAT WILL HANDLE THE RECEIVING OF THE UPDATED MODEL FROM CLIENT client
                updated_model_listener = threading.Thread(target=receive_model_from_client, args=(self, _socket))
                updated_model_listener.start()
                print(f"Getting local model from client: {client['id']}")
            except socket.error as e:
                continue
                # print(f"Server {self.port_no} error SENDING W to client {client['id']}")
                # print(f"ERROR {e}")

    def compute_new_global_model(self, selected_clients, total_size):
        """
        Computes the new global model starting from the models of the selected clients (depending on the subsampling)
        :param selected_clients: Selected clients as a list, depends on the subsampling mode
        :param total_size: total size of the data of the alive clients, used in the computation of the new weight
        """
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for client in selected_clients:
            client_model = self.get_model(client)
            if client_model:
                for server_param, user_param in zip(self.model.parameters(), client_model.parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * client["data_size"] / total_size

    def compute_global_averages(self):
        """
        Computes the global average accuracy and the global average training loss at each communication round, then empties the two lists so they are ready for the next round.
        """
        self.global_average_training_loss = sum(self.round_losses) / len(self.round_losses)
        self.global_average_accuracy = sum(self.round_accuracies) / len(self.round_accuracies)

        # SET THE ROUND LOSSES AND ACCURACIES TO AN EMPTY ARRAY
        self.round_losses = []
        self.round_accuracies = []

    def log_global_averages(self, communication_round):
        """
        Logs global averages to a log file
        """
        with open("server_average_loss_accuracy.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [communication_round, float(self.global_average_accuracy), float(self.global_average_training_loss)])

    def subsample_clients(self):
        """
        Subsample clients if the subsampling option is set to True
        :return: the list of the sampled clients
        """
        match len(self.clients):
            case 0:
                print("No alive clients...")
                return list([])
            case 1:
                return list([self.clients[0]])
            case 2:
                return list([self.clients[0], self.clients[1]])
        randomly_selected = random.choices(self.clients, k=2)
        return randomly_selected

    def get_model(self, client):
        """
        Given a client, returns the model sent by the client
        :param client: client dictionary { "id": int, "port_no": int, "data_size": int, "model_sent": int}
        :return: client's model as np.array
        """
        client_id = client["id"]
        model = False
        if self.clients_models.keys().__contains__(client_id):
            model = self.clients_models[client_id]
        return model

    def get_total_size(self):
        """
        Get total data size of the currently alive clients
        """
        total_size = 0
        for client in self.clients:
            total_size += client["data_size"]
        return total_size


server = Server()
server.run()
