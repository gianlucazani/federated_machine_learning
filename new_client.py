import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import socket
import _pickle
import sys

HOST = "127.0.0.1"


# -------------------------------------------------------------------------------------
# -------------------------------------- METHODS --------------------------------------
# -------------------------------------------------------------------------------------


def get_data(_id):
    """
    Retrieves data from the right file depending on the client id
    :param _id: id of the client retrieving the data
    :return: train and tests sets as tensors
    """
    train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(_id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(_id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0'][
        'y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples


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


# -------------------------------------------------------------------------------------
# -------------------------------------- CLASSES --------------------------------------
# -------------------------------------------------------------------------------------

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


class Client:
    def __init__(self, batch_size):

        self.id = sys.argv[1]
        self.port_no = int(sys.argv[2])
        self.optimization_method = sys.argv[3]  # 0 for GD, 1 for Mini-Batch GD

        self.server_port_no = 6000  # fixed to 6000

        # LOG FILES
        self.log_file = os.path.join("./", "client" + str(self.id) + "_log.txt")

        # clean file before starting
        with open(self.log_file, 'w') as f:
            f.write("")
        with open(f'evaluation_log_{self.id}.csv', "w") as f_eval:
            f_eval.write("")
            writer = csv.writer(f_eval)
            writer.writerow(["client_id", "communication_round", "loss", "accuracy"])

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(
            self.id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.full_dataset = DataLoader(self.train_data,
                                       self.train_samples)
        self.batched_dataset = DataLoader(self.train_data,
                                          batch_size)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.bind((HOST, int(self.port_no)))

        self.model = MCLR()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=0.01)  # here the optimizer is set to be stochastic gradient descent
        self.loss = nn.NLLLoss()

    def set_parameters(self, model):
        """
        Updates the old model's parameters with the new one, which will be the global model sent by the server at each communication round
        :param model: the new model
        """
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def run(self):
        rounds_left = 100

        # SEND HANDSHAKE FOR CONNECTING TO SERVER
        self.handshake()

        # WRITE HEADER IN THE EVALUATION FILE
        f_eval = open(f'evaluation_log_{self.id}.csv', "a+")
        writer = csv.writer(f_eval)

        # START LISTENING FOR GLOBAL MODEL, EVALUATE IT, TRAIN IT, SEND BACK
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
                    global_model = received_global_model['model']

                    # UPDATE CLIENT'S MODEL WITH THE GLOBAL ONE
                    self.set_parameters(global_model)

                    # CALCULATE LOSS AND TRAIN
                    training_loss = self.train()
                    print("Local training...")
                    print(f"Training loss: {training_loss}")
                    # TEST THE MODEL
                    testing_accuracy = self.test()
                    print(f"Testing accuracy: {testing_accuracy}")

                    # CREATE UPDATE PACK TO SEND BACK TO SERVER
                    updated_weights = {
                        'model': self.model,
                        'id': str(self.id),
                        'training_loss': training_loss,
                        'testing_accuracy': testing_accuracy
                    }

                    # SEND UPDATED MODEL BACK TO SERVER
                    print("Sending back new global model")
                    conn.sendall(_pickle.dumps(updated_weights))
                    rounds_left = received_global_model['rounds_left']

                    try:
                        # WRITE TO LOG FILE
                        f.write(f'I am client {self.id} \n')
                        f.write(f'Receiving new global model \n')
                        f.write(f'Local training... \n')
                        f.write(f'Training loss: {training_loss} \n')
                        f.write(f'Testing accuracy: {testing_accuracy} \n')

                        # WRITE TO EVALUATE LOG FILE
                        writer.writerow([self.id, 100 - rounds_left, float(training_loss), testing_accuracy])
                    except IOError as e:
                        print(f"Client {self.id} error writing to log file")
                        print(f"ERROR: {e}")
            except socket.error as e:
                print(f"Client {self.id} error while communicating with server")
                print(f"ERROR: {e}")
                continue
        f_eval.flush()
        f_eval.close()

    def train(self):
        """
        Performs training of the model by running two epochs. The optimization will change depending on the optimization method chosen at start
        :return: the loss of the model over the training set
        """
        self.model.train()
        for epoch in range(2):
            self.model.train()
            if self.optimization_method == "0":
                dataset_to_use = self.full_dataset
            else:
                dataset_to_use = self.batched_dataset
            for i, (X, y) in enumerate(dataset_to_use):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        """
        Tests the model over the client's test set
        :return: the model accuracy on the testing set
        """
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
        return test_acc

    def handshake(self):
        """
        Sends handshake to the server
        """
        try:
            print("Sending handshake")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, self.server_port_no))
                packet = {
                    'id': str(self.id),
                    'port_no': int(self.port_no),
                    'data_size': self.X_train.shape[0],
                    'model_sent': 0
                }
                s.sendall(_pickle.dumps(packet))
        except socket.error as e:
            print(f"Client {self.id} cannot send handshake")
            print(f"ERROR: {e}")
            pass


client = Client(5)
client.run()
