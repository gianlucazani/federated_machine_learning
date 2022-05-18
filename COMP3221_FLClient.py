import sys

HOST = "127.0.0.1"


class Client:
    def __init__(self):
        self.id = sys.argv[1]
        self.port_no = sys.argv[2]
        self.server_port_no = 6000  # fixed to 6000
        self.optimization_method = sys.argv[3]  # 0 for GD, 1 for Mini-Batch GD

    def run(self):
        pass


client = Client()
client.run()



