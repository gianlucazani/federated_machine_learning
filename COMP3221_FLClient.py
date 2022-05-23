import random
import sys
import os

import numpy as np
import json
import socket
import threading
import _pickle

HOST = "127.0.0.1"
np.seterr(all='raise')


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


class Client:
    def __init__(self, batch_size):
        self.id = sys.argv[1]
        self.port_no = int(sys.argv[2])
        self.optimization_method = sys.argv[3]  # 0 for GD, 1 for Mini-Batch GD

        self.server_port_no = 6000  # fixed to 6000

        # LOG FILE
        self.log_file = os.path.join("./", "client" + str(self.id) + "_log.txt")
        # clean file before starting
        with open(self.log_file, 'w') as f:
            f.write("client_id, communication_round, loss, accuracy")

        # RETRIEVE DATASET
        self.batch_size = batch_size
        self.X_train = np.array([], dtype=np.float128)
        self.Y_train = np.array([], dtype=np.float128)
        self.X_test = np.array([], dtype=np.float128)
        self.Y_test = np.array([], dtype=np.float128)
        self.read_dataset()

        # SOCKET
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.bind((HOST, int(self.port_no)))
        handshake_thread = threading.Thread(target=self.handshake())
        handshake_thread.start()

    def run(self):
        rounds_left = 100

        while rounds_left > 0:
            try:
                self.client_socket.listen()
                with open(self.log_file, 'a+') as f:

                    # RECEIVE GLOBAL MODEL
                    print(f'I am client {self.id}')
                    conn, addr = self.client_socket.accept()
                    print(f'Receiving new global model')
                    received = receive_all(conn)
                    received_global_model = _pickle.loads(received)
                    # print(f"-- Received a packet with dimension: {sys.getsizeof(received)} --")
                    W = received_global_model['W']

                    # CALCULATE TRAINING LOSS (loss is calculated over the entire set even with mini-batch)
                    try:  # sometimes the softmax function raises "overflow encountered in exp", because the exponent is too big
                        training_loss = self.softmax_loss(self.X_train, self.Y_train, W)
                    except RuntimeWarning as e:
                        continue
                    print(f'Training loss: {training_loss}')

                    # PREDICT USING THE GLOBAL MODEL AND TEST ACCURACY (accuracy is calculated over the entire set even with mini-batch)
                    y_predictions = self.predict(W, self.X_test)
                    testing_accuracy = self.accuracy(y_predictions, self.Y_test)
                    print(f'Testing accuracy: {testing_accuracy}')

                    # TRAIN THE MODEL WITH GD or MINI-BATCH GD
                    print(f'Local training...')
                    if self.optimization_method == "1":
                        X_batch_train, y_batch_train = self.get_batch()
                        W, loss_hist = self.softmax_fit(X_batch_train, y_batch_train, W)
                    elif self.optimization_method == "0":
                        W, loss_hist = self.softmax_fit(self.X_train, self.Y_train, W)
                    updated_weights = {
                        'W': W,
                        'id': str(self.id)
                    }

                    # SEND UPDATED MODEL BACK TO SERVER
                    # print(f"-- Sending a packet with dimension: {sys.getsizeof(_pickle.dumps(updated_weights))} -- ")
                    conn.sendall(_pickle.dumps(updated_weights))
                    rounds_left = received_global_model['rounds_left']

                    # WRITE TO FILE
                    # we might choose to write something different to the file, maybe something easier to be read when performing final evaluation
                    # we could print data in a fancier way (maybe .csv)
                    # the text about what the client is doing is only left to be printed at terminal but not in the log file

                    # Writing is moved down here for catching exception separately
                    try:
                        f.write(f"{self.id}, {100 - rounds_left}, {training_loss}, {testing_accuracy}\n")
                    except IOError as e:
                        print(f"Client {self.id} error writing to log file")
                        print(f"ERROR: {e}")
            except socket.error as e:
                print(f"Client {self.id} error while communicating with server")
                print(f"ERROR: {e}")
                continue

    def get_batch(self):
        batch_observations_indexes = random.sample(range(0, self.X_train.shape[0]), self.batch_size)
        X_batch_train = self.X_train[batch_observations_indexes, :]
        y_batch_train = self.Y_train[batch_observations_indexes,]
        return X_batch_train, y_batch_train

    def handshake(self):
        try:
            print("Sending handshake")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, self.server_port_no))
                packet = {
                    'id': str(self.id),
                    'port_no': int(self.port_no),
                    'data_size': self.X_train.shape[0]
                }
                s.sendall(_pickle.dumps(packet))
        except socket.error as e:
            print(f"Client {self.id} cannot send handshake")
            print(f"ERROR: {e}")
            pass

    def read_dataset(self):
        train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(self.id) + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(self.id) + ".json")
        train_data = {}
        test_data = {}
        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data'])
        self.X_train, self.Y_train, self.X_test, self.Y_test = train_data['0']['x'], train_data['0']['y'], \
                                                               test_data['0']['x'], test_data['0']['y']
        self.X_train = np.array(self.X_train, dtype=np.float128)
        self.Y_train = np.array(self.Y_train, dtype=np.float128)
        self.Y_train = self.Y_train.astype(int)
        self.X_test = np.array(self.X_test, dtype=np.float128)
        self.Y_test = np.array(self.Y_test, dtype=np.float128)
        self.X_train = np.concatenate((self.X_train, np.ones((self.X_train.shape[0], 1))), axis=1)
        self.X_test = np.concatenate((self.X_test, np.ones((self.X_test.shape[0], 1))), axis=1)

    def softmax(self, X, W):
        Z = X.dot(W)
        try:
            e_Z = np.exp(Z)
            A = e_Z / e_Z.sum(axis=1, keepdims=True)
        except Exception as e:
            print("")
        return A

    def softmax_loss(self, X, y, W):
        A = self.softmax(X, W)
        id0 = np.arange(X.shape[0])
        return -np.mean(np.log(A[id0, y]))

    def softmax_grad(self, X, y, W):
        prediction = self.softmax(X, W)
        xid = range(X.shape[0])
        prediction[xid, y] -= 1
        return X.T.dot(prediction) / X.shape[0]

    def softmax_fit(self, X, y, W, lr=0.35, epochs=2):
        ep = 0
        loss_hist = [self.softmax_loss(X, y, W)]  # store history of loss
        while ep < epochs:
            ep += 1
            W -= lr * self.softmax_grad(X, y, W)
            loss_hist.append(self.softmax_loss(X, y, W))
        return W, loss_hist

    def accuracy(self, y_predictions, y):
        diff = y_predictions - y
        n_zeros = np.count_nonzero(diff == 0)
        np.asanyarray(n_zeros)
        accuracy = n_zeros / len(y)
        return accuracy

    def predict(self, W, X):
        A = self.softmax(X, W)
        return np.argmax(A, axis=1)


client = Client(20)
client.run()
