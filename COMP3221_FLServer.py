import _pickle
import random
import socket
import sys
import threading
import time

import numpy as np

HOST = "127.0.0.1"


def receive_all(sock):
    """
    Due to high dimensional packets (about 60k bytes) exchanged between server and client, we need to make the read of the received message incremental,
    with steps equal to the maximum supported size: 4096 bytes.
    :param sock: socket to receive from
    :return: the whole data in bytes
    """
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
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

        np.random.seed(123)
        self.W = np.random.randn(785, 10)

        # print(sys.getsizeof(self.W))

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, int(self.port_no)))

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

    def federated_learning(self):
        for communication_round in range(50):
            #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # BROADCAST GLOBAL MODEL AND RECEIVE UPDATED ONE FROM CLIENTS
            self.send_and_receive_model(50 - 1 - communication_round)
            # # BROADCAST GLOBAL MODEL TO ALL THE CLIENTS
            # self.broadcast_W(5 - 1 - communication_round)
            #
            # # RECEIVE LOCAL MODELS FROM CLIENTS
            # self.server_socket.listen()
            # for i in range(len(self.clients)):
            #     client_socket, address = self.server_socket.accept()
            #     received = receive_all(client_socket)
            #     received_clients_model = _pickle.loads(received)
            #     client_id = received_clients_model["id"]
            #     clients_model = received_clients_model["W"]
            #     self.clients_models[client_id] = clients_model

            # SELECT WHICH MODELS TO COMBINE FOR THE NEW GLOBAL MODEL
            if self.client_subsampling:
                selected_clients = self.subsample_clients()
            else:
                selected_clients = self.clients

            # GET TOTAL DATA SIZE (sum of data sizes of the alive clients)
            total_size = self.get_total_size()

            # COMPUTE NEW GLOBAL MODEL
            self.W = self.compute_new_global_model(selected_clients, total_size)

            # CLEAN MODELS DICTIONARY FOR NEXT ITERATION
            self.clients_models = dict()

    def send_and_receive_model(self, rounds_left):
        for client in self.clients:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _socket:
                try:
                    # PREPARE MODEL PACKAGE
                    package = {
                        "W": self.W,
                        "rounds_left": rounds_left
                    }
                    message = _pickle.dumps(package)
                    # print(sys.getsizeof(message))

                    # SEND MODEL
                    _socket.connect((HOST, int(client["port_no"])))
                    _socket.sendall(message)

                    # RECEIVE UPDATED MODEL PACKAGE
                    received = receive_all(_socket)
                    received_clients_model = _pickle.loads(received)
                    client_id = received_clients_model["id"]
                    clients_model = received_clients_model["W"]
                    self.clients_models[client_id] = clients_model

                    print(f"Sent global model. Rounds left: {rounds_left}")
                except socket.error as e:
                    print(f"Server {self.port_no} error SENDING W to client {client['id']}")
                    print(f"ERROR {e}")

    # def broadcast_W(self, rounds_left: int):
    #     """
    #     Sends the current global model W to all the connected clients
    #     :param _socket: socket used for sending and receiving models
    #     :param rounds_left: number of global communication rounds left (used by clients for stopping themselves)
    #     """
    #     for client in self.clients:
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _socket:
    #             try:
    #                 packet = {
    #                     "W": self.W,
    #                     "rounds_left": rounds_left
    #                 }
    #                 message = _pickle.dumps(packet)
    #                 print(sys.getsizeof(message))
    #                 _socket.connect((HOST, int(client["port_no"])))
    #                 _socket.sendall(message)
    #                 print(f"Sent global model. Rounds left: {rounds_left}")
    #             except socket.error as e:
    #                 print(f"Server {self.port_no} error SENDING W to client {client['id']}")
    #                 print(f"ERROR {e}")

    def compute_new_global_model(self, selected_clients, total_size):
        result = np.zeros_like(self.get_model(selected_clients[0]))  # init the result to the same shape as the model but filled with 0, necessary for performing sum
        for client in selected_clients:
            model = self.get_model(client)
            model = np.array(model)
            coefficient = client["data_size"] / total_size
            result += coefficient * model
        return result

    def subsample_clients(self):
        """
        Subsample clients if the subsampling option is set to True
        :return: the list of the subsampled clients
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
        model = np.array(self.clients_models[client_id])
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
