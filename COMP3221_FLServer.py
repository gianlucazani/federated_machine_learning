import _pickle
import random
import socket
import sys
import threading
import time

import torch
import torch.nn as nn  # neural network
import torch.nn.functional as F  # like the sigmoid, softmax, ...

HOST = "127.0.0.1"


class MCLR(nn.Module):

    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)
        self.fc1.weight.data = torch.randn(self.fc1.weight.size()) * .01

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor.
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


def receive_model_from_client(server, _socket):
    try:
        received = receive_all(_socket)
        received_clients_model = _pickle.loads(received)
        client_id = received_clients_model["id"]
        clients_model = received_clients_model["W"]
        server.clients_models[client_id] = clients_model
        _socket.close()
    except socket.error as e:
        pass


def receive_all(sock):
    """
    Due to high dimensional packets (about 60k bytes) exchanged between server and client, we need to make the read of the received message incremental,
    with steps equal to the maximum supported size: 4096 bytes.
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


class HandshakeThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self) -> None:
        while server.is_alive():
            self.server.server_socket.listen()
            conn, addr = self.server.server_socket.accept()
            received = conn.recv(32768)
            client = _pickle.loads(received)
            print(f"Client connected: {client}")
            self.server.clients.append(client)


class Server:
    def __init__(self):
        self.port_no = sys.argv[1]  # fixed to 6000
        self.client_subsampling = sys.argv[2]  # 0 for use all the clients' models, 1 use only 2 chosen randomly
        if self.client_subsampling == "0":
            self.client_subsampling = False
        else:
            self.client_subsampling = True
        self.alive = False

        self.clients = list()  # list of objects { "id": int, "port_no": int, "data_size": int }
        self.clients_models = dict()  # (key = ID, value = model)

        self.model = MCLR()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, int(self.port_no)))
        self.communication_rounds = 100

    def run(self):
        self.alive = True
        handshake_thread = HandshakeThread(self)
        self.server_socket.listen()
        conn, addr = self.server_socket.accept()
        received = conn.recv(4096)
        client = _pickle.loads(received)
        self.clients.append(client)
        print(f"Client connected: {client}")
        handshake_thread.start()
        time.sleep(10)
        self.federated_learning()
        print("finished learning: evaluating model...")
        counter=0
        total_loss = 0
        total_accuracy = 0
        for client in self.clients:
            file_path = 'evaluation_log_'+client['id']+'.csv'
            with open(file_path, 'r') as f:
                for line in f:
                    counter+=1
                    line = line.split(',')
                    try:
                        total_loss += float(line[2])
                        total_accuracy += float(line[3])
                    except:
                        continue
        average_loss = total_loss/(counter-len(self.clients))
        average_accuracy = total_accuracy/(counter-len(self.clients))
        print(f'Average Loss is: {average_loss}')
        print(f'Average Accuracy is: {average_accuracy}')


    def federated_learning(self):
        """
        Runs the federated learning algorithm
        """
        for communication_round in range(self.communication_rounds):
            print(f"Global Iteration: {communication_round}")
            print(f"Total number of clients: {len(self.clients)}")
            self.broadcast_global_model(self.communication_rounds - 1 - communication_round)

            # CHECK THAT EVERY CLIENT SENT THE UPDATED MODEL
            received_all_models = self.check_received_all_models()
            if not received_all_models:
                self.handle_dead_clients()

            if self.client_subsampling:
                selected_clients = self.subsample_clients()
            else:
                selected_clients = self.clients

            # GET TOTAL DATA SIZE (sum of data sizes of the alive clients)
            total_size = self.get_total_size()

            # COMPUTE NEW GLOBAL MODEL
            self.compute_new_global_model(selected_clients, total_size)

            # CLEAN MODELS DICTIONARY FOR NEXT ITERATION
            self.clients_models = dict()

    def handle_dead_clients(self):
        for client in self.clients:
            if not self.clients_models.keys().__contains__(client["id"]):
                # client dead and will be removed from the clients list
                print(f"------- CLIENT {client['id']} DIED --------")
                self.clients.remove(client)

    def check_received_all_models(self) -> bool:
        """
        Checks for a maximum of 10 seconds if all the models have been received by the server
        if this happens before 10 seconds, then it means that all the alive clients sent their model
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
        Sends global model to client and receives models from clients one by one
        :param rounds_left: rounds left for completing the federated training, will be sent to clients to let them know when to stop themselves
        """
        print("Broadcasting global model")
        for client in self.clients:
            _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # PREPARE MODEL PACKAGE
                package = {
                    "W": self.model,
                    "rounds_left": rounds_left
                }
                message = _pickle.dumps(package)

                # SEND MODEL
                _socket.connect((HOST, int(client["port_no"])))
                _socket.sendall(message)

                # START NEW THREAD THAT WILL HANDLE THE RECEIVING OF THE UPDATED MODEL FROM CLIENT client
                updated_model_listener = threading.Thread(target=receive_model_from_client, args=(self, _socket))
                updated_model_listener.start()
                print(f"Getting local model from client: {client['id']}")
            except socket.error as e:
                print(f"Server {self.port_no} error SENDING W to client {client['id']}")
                print(f"ERROR {e}")

    def compute_new_global_model(self, selected_clients, total_size):
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for client in selected_clients:
            for server_param, user_param in zip(self.model.parameters(), self.get_model(client).parameters()):
                server_param.data = server_param.data + user_param.data.clone() * client["data_size"] / total_size

    def subsample_clients(self):
        """
        Subsample clients if the subsampling option is set to True
        :return: the list of the sampled clients
        """
        match len(self.clients):
            case 1:
                return list([self.clients[0]])
            case 2:
                return list([self.clients[0], self.clients[1]])
        randomly_selected = random.choices(self.clients, k=2)
        return randomly_selected

    def get_model(self, client):
        """
        Given a client, returns the model sent by the client
        :param client: client dictionary { "id": int, "port_no": int, "data_size": int }
        :return: client's model as np.array
        """
        client_id = client["id"]
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

    def is_alive(self) -> bool:
        return self.alive


server = Server()
server.run()
