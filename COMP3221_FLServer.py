import sys
import threading 

class Server:
    def __init__(self):
        self.port_no = sys.argv[1]  # fixed to 6000
        self.subsample_clients = sys.argv[2]  # 0 for use all the clients' weights, 1 use only 2 chosen randomly

    def run(self):

        start_wss_thread = threading.Thread(target=self.start_websocket_server)
        start_wss_thread.start()
    

    def start_websocket_server(self):
        """
        This is the method that starts the websocket server. The handling clients thread is called when a client connects to the websocket server. 
        """
        self.server.listen()
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, ))
            thread.start()

server = Server()
server.run()
