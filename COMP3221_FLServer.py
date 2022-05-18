HOST = "127.0.0.1"


class Client:
    def __init__(self, id, port_no, server_port_no):
        self.id = id
        self.port_no = port_no
        self.server_port_no = server_port_no

    def run(self):
        pass


client = Client()
client.run()
