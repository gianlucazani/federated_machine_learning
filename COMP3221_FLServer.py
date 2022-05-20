import _pickle
import socket
import sys
import threading
import time

import numpy as np

HOST = "127.0.0.1"
CLIENT_HOST = "172.20.10.6"


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
            self.server.clients.update(client)

class Server:
    def __init__(self):
        self.port_no = sys.argv[1]  # fixed to 6000
        self.subsample_clients = sys.argv[2]  # 0 for use all the clients' weights, 1 use only 2 chosen randomly
        self.clients = dict()  # (key = ID, value = PORT)
        self.clients_weights = dict()  # (key = ID, value = weight)
        self.alive = False
        self.W = np.random.randn(785, 10)
        print(sys.getsizeof(self.W))
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((SERVER_HOST, int(self.port_no)))

    def run(self):
        self.alive = True
        handshake_thread = HandshakeThread(self)
        self.server_socket.listen()
        conn, addr = self.server_socket.accept()
        received = conn.recv(131072)
        client = _pickle.loads(received)
        self.clients.update(client)
        print("first client connected!!")
        handshake_thread.start()
        time.sleep(3)
        self.federated_learning()

    def is_alive(self) -> bool:
        return self.alive

    def federated_learning(self):
        for communication_round in range(5):
            self.broadcast_W(100 - communication_round)
            self.server_socket.listen()
            conn, addr = self.server_socket.accept()
            received = conn.recv(131072)
            new_weight = _pickle.loads(received)
            self.W = new_weight["W"]
            # for client in range(len(self.clients.keys())):
            #     self.server_socket.listen()
            #     conn, addr = self.server_socket.accept()
            #     received = conn.recv(4096)
            #     new_weight = _pickle.loads(received)
            #     # self.clients_weights[]
            print(self.W)

    def broadcast_W(self, rounds_left: int):
        for client_id, destination_port in self.clients.items():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    packet = {
                        "W": self.W,
                        "rounds_left": rounds_left
                    }
                    message = _pickle.dumps(packet)
                    print(sys.getsizeof(message))
                    s.connect((CLIENT_HOST, int(destination_port)))
                    s.sendall(message)
                except socket.error as e:
                    print(f"Server {self.port_no} error SENDING W to client {client_id}")
                    print(f"ERROR {e}")

server = Server()
server.run()
