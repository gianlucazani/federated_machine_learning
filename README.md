# COMP3221 Assignment3: Report
<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170199199-2f7854e2-7829-45c8-9321-0d2dd4635198.png" width="700px"/>
</p>
The assignment topic is the realization of a Federated Training network where a Server aggregates the models given by five Clients which train the global model on a subset of the original data. 
## Information

The implementation of the Federated Training satisfies all the assignment requirements:
<ul>
  <li>
    Server broadcasts the global model to each client, and all the clients will be training the model in parallel.
  </li>
  <li>
    Server prints at terminal information about the Federated Training that is taking place.
  </li>
  <li>
    Each clients outputs at terminal information about loss and accuracy of the model.
  </li>
  <li>
    Each client will write to a file the losses and accuracies for each communication round.
  </li>
  <li>
    Server aggregates the models received by the clients based on the subsampling mode that has been chosen when starting the server: if subsamplin is active, only two out of the alive clients will be chosen for their models aggregation. Otherwise all of the alive clients will be choosen.
  </li>
  <li>
    Clients can run the training both using standard Gradient Descent and Mini-Batch Gradient Descent.
  </li>
    Each client can join the network at any moment during the execution of the training and will start contrinuting to it.
  <li>
    At the end of the training the server will print at terminal the overall average accuracy and average loss, as well as the overall training time
  </li>
  <li>
    
  </li>
  <li>
    
  </li>
  <li>
    
  </li>
  <li>
    [BONUS] Client failure is handled.
  </li>
</ul>

## Environment and Dependencies

This program runs on:

```
python 3.10.4
```
The program needs the following packages:
```
torch 1.11.0
numpy 1.22.3
```
## Usage

This section will explain how to use the program and see as the network behaviour satisfies requirements.

### Starting the server
As stated in the assignment sheet, the server is started by running the following shell command:
```
python3 COMP3221_FLServer.py <Port-Server> <Sub-Client>
```
Where ```Port-Server``` has to be ```6000``` by default and ```Sub-Client``` can be both ```0``` (all alive clients are selected for aggregation) or ```1``` (two out of the number of alive clients are selected for agregation).

### Starting a client
As stated in the assignment sheet, a client is started by running the following shell command:
```
python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
```
Where ```Client-id``` should be an integer positve value, ```Opt-Method``` could be both ```0``` (for Gradient Descent) or ```1``` (for Mini-Batch Gradient Descent).

## Implementation

The graph below shows how the flow of execution achieves the requirements:

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170196856-8c37f6de-2ee6-4b07-b5d8-151d2761045e.png" width="800px"/>
</p>

Where the meaning of the acronyms are:
<ul>
  <li>
    <b>GAAn</b>: Global Average Accuracy of the n-th Model (or at (n - 1)-th round)
  </li>
  <li>
    <b>GATLn</b>: Global Average Training Loss of the n-th Model (or at (n - 1)-th round)
  </li>
  <li>
    <b>LTLnm</b>: Local Trianing Loss while training the n-th Model (or at (n - 1)-th round) on the m-th client's training set
  </li>
  <li>
    <b>GMALDnm</b>: Global Model Accuracy on Local Data of the n-th Model (or at (n - 1)-th round) on the m-th client's test set
  </li>
</ul>

Note that the shema above refers to an execution of two communication rounds with two clients. Moreover, the actions taken by the server and the clients are more than the ones written in the schema, which is a simplified and essential version of the program. 

We will now explore more in depth how the execution works.

### Server's point of view

The server will execute in this way (in order):

<ul>
  <li>
    Generate initial random model (weights vector) at initialization
  </li>
  <li>
    Waits for handshake from the first client
  </li>
  <li>
    Once the first handshake is received, waits for 30 seconds for other handshakes before starting the federated training. Clients will be able to join the network at any time, because the server keeps listening for handskakes thanks to the HandshakeThread
  </li>
  <li>
    After waiting the 30 seconds, the server will start the federated training
    <ul>
      <li>
        Broadcast global model to all alive clients
      </li>
      <li>
        Wait for all the responses with updated models and statistics about the global model just sent
      </li>
      <li>
        If some of the supposingly alive clients don't send the model, find out which died and remove it from the alive clients list
      </li>
      <li>
        Save clients' tested accuracies and losses, will be used both for calculating average accuracy and loss at each round and for calculating the         overall final average accuracy and loss when the training ends.
      </li>
      <li>
        Aggregate selected clients' models with a weighted average, in this way the server generate the new global model
      </li>
      <li>
        Repeat the process until the communication rounds end
      </li>
    </ul>
  </li>
</ul>

### Client's point of view

The client execution is structured as follows:

<ul>
  <li>
    Load data from the local dataset and save them in data structures, which will store training and test data separately
  </li>
  <li>
    Send handshake to the server in the format specified in the assignment sheet
  </li>
  <li>
    Upon reception of the packet from the server, the client will execute in this way:
    <ul>
      <li>
        Print at terminal the statistics about global average loss and accuracy of the received model
      </li>
      <li>
        Test the received model on the local test set
      </li>
      <li>
        Train the model on the local training set and get the training loss
      </li>
      <li>
        Test the newly trained model on the test dataset
      </li>
      <li>
        Send back to server the updated model together with the training loss and the accuracy on the test data of the received global model
      </li>
      <li>
        While printing at terminal, save the same output to the client's log file
      </li>
      <li>
        Save communication round, global model average accuracy, global model average training loss, global model accuracy on local data, training loss and accuracy of the newly trained model in a .csv file used for evaluation purposes.
      </li>
      <li>
        Check how many rounds are left and if they're 0, stop. If some other rounds are left, keep listening for new global model from the server.
      </li>
    </ul>
  </li>
</ul>

## Results and Examples

In this section examples reuslts and comparison are explained. 

### Comparison between clients with same settings

By running a full training with no subsampling and Mini-Batch GD with batch size equal to 20, this is how the clients performed one against the other:

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170224648-4a410107-0b81-4a59-b19b-da0961963464.png" width="550px"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170224948-64a8eada-8b27-43c4-8ebc-514a4621e5b9.png" width="550px"/>
</p>

While the overall accuracy and loss seen by the server at each round has the following behaviour:

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170225987-ab5bad63-b59f-4437-bd72-e3a9d038803f.png" width="550px"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170226244-0a1a1907-5f46-47c3-a590-edd3f753f632.png" width="550px"/>
</p>

### Comparison between different batch sizes in Mini-Batch GD

In this section we will be exploring how the different batch sizes perform over the communication round. The setting of the testing will not vary except the batch size, we will run the training with all five clients and no subsampling mode:

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170230575-66ced7a1-885b-4abf-9eea-8a4408217c0f.png" width="550px"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170230785-ceb35340-9df3-4a01-9c4a-8737e38da9cf.png" width="550px"/>
</p>


