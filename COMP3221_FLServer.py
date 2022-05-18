import sys


class Server:
    def __init__(self):
        self.port_no = sys.argv[1]  # fixed to 6000
        self.subsample_clients = sys.argv[2]  # 0 for use all the clients' weights, 1 use only 2 chosen randomly

    def run(self):
        pass


server = Server()
server.run()
