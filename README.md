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
    <b>[BONUS]</b> Client failure is handled.
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

### Simulate client's failure

To simulate client failure it is sufficient to kill the process of the client we want to make fail. This could be done with the ```ctrl+c``` command at terminal.

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

The server output at terminal will be as follows (5 clients connected):

```
Global Iteration: 1
Total number of clients: 5
Broadcasting global model
Getting local model from client: 1
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Global model average accuracy: 13.30%
Local average training loss:v 0.01945
Global Iteration: 2
Total number of clients: 5
Broadcasting global model
Getting local model from client: 1
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Global model average accuracy: 88.30%
Local average training loss:v 0.01735
.
.
.
Global Iteration: 100
Total number of clients: 5
Broadcasting global model
Getting local model from client: 1
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Global model average accuracy: 92.22%
Local average training loss:v 0.00752
finished learning: evaluating model...
Average Loss is: 0.008229239843785763
Average Accuracy is: 0.914145193554461
Training time: 115.82871389389038
```

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

### Files produced during execution

#### Client's files

During execution, all the clients will produce two files.<br>
The first is called ```client<client-id>_log.txt``` and will store the the output of the client ```<client-id>``` during execution, the content of the file will be exactly the same as the output of the client at terminal. In this file you will find a descriptive sequence of action taken by the client, for example client 1 will write to its ```client1_log.txt``` something like:
```
I am client 1 
Receiving new global model
Global Communication round: 1
Initial training round, no global accuracy or training loss
Global model accuracy tested on local data: 0.08020231127738953
Local training... 
Local Training loss: 0.015210394747555256 
Local model Testing accuracy: 0.9696531891822815 
Sending back new global model
I am client 1 
Receiving new global model 
Global Communication round: 2
Global Average Accuracy 0.08030781745910645
Global Average Training Loss 0.01812969520688057
Global model accuracy tested on local data: 0.7933526039123535
Local training... 
Local Training loss: 0.013666236773133278 
Local model Testing accuracy: 0.9674855470657349 
Sending back new global model
.
.
.
I am client 1 
Receiving new global model 
Global Communication round: 100
Global Average Accuracy 0.9222921967506409
Global Average Training Loss 0.0074506038799881935
Global model accuracy tested on local data: 0.897398829460144
Local training... 
Local Training loss: 0.00599539652466774 
Local model Testing accuracy: 0.9559248685836792 
Sending back new global model
```
Where each section starting with ```I am client 1``` corresponds to one communication round.

The other file is called ```evaluation_log_<client-id>.csv``` and will store just the numeric values obtained from the various testing and training perfromed during each communication round. The file built from client 1, ```evaluation_log_1.csv```, will store something like:
```
client_id,communication_round,local_training_loss,local_model_testing_accuracy,global_model_accuracy
1,1,0.015210394747555256,0.9696531891822815,0.08020231127738953
1,2,0.013666236773133278,0.9674855470657349,0.7933526039123535
.
.
.
1,100,0.00599539652466774,0.9559248685836792,0.897398829460144
```
Where ```local_training_loss``` is the loss computed over the training of the received global model on the local data, ```local_model_testing_accuracy``` is the accuracy of the newly trained model on the local test data and ```global_model_accuracy``` is the accuracy of the received global model on the local test data before the training.

The purpose of this file is to make it easier to evaluate the performance of the client after the training, see values as tables, plot values and similar oeprations. With the ```client<client-id>_log.txt``` such operation would be much more tricky and error prone.

#### Server's files

The server, as the client, will write data to two different log files.
The fist is ```server_overall_data_log.csv``` where all the clients' local training losses and accuracies received during the whole training will be stored. This means that if 5 clients are alive and the training is done for 100 communication rounds, this file will contain 5 * 100 = 500 rows. The logged data in this file will be something like:
```
losses,accuracies
0.0003177614707965404,0.1240951418876648
0.00013672371278516948,0.9824198484420776
8.666139910928905e-05,0.9813857078552246
.
.
.
1.1920928244535389e-07,0.9813857078552246
1.1920928244535389e-07,0.9813857078552246
```
Where ```losses``` are the local losses sent by clients at each round and same for ```accuracies```. The purpose of this file is to calculate the overall training loss and accuracy after the whole training has ended.

The second file is called ```server_average_loss_accuracy.csv``` and will store, per each communication round, the average local training loss and accuracy of the alive clients. The content of the file will look like:
```
communication_round,global_average_accuracy,global_average_loss
1,0.09974871575832367,0.008109461516141891
2,0.9334225356578827,0.006449408829212189
.
.
.
100,0.9509825110435486,0.0018738580401986837
```
Where ```global_average_accuracies``` is the average of the clients' global model's accuracy on local test data and ```global_average_loss``` is the average of clients' local training loss.
The purpose of this file is the evaluation of these averaged values over communication rounds.

## Execution examples

### Client's failure

The client failure is handled by the server. In the following example all the five clients where alive before client 2 fails. This is what the server prints at terminal:

```
Global Iteration: 11
Total number of clients: 5
Broadcasting global model
Getting local model from client: 2
Getting local model from client: 1
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Global model average accuracy: 92.36%
Local average training loss:v 0.00952
Global Iteration: 12
Total number of clients: 5
Broadcasting global model
Getting local model from client: 2
Getting local model from client: 1
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
------- CLIENT 1 DIED --------
Global model average accuracy: 93.19%
Local average training loss:v 0.00964
Global Iteration: 13
Total number of clients: 4
Broadcasting global model
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Global model average accuracy: 93.61%
Local average training loss:v 0.00883
```
The server detects the failure and after printing which client 1 has failed, will update the alive clients list.

### Client joining later

If a client joins later the network, it will start contributing to the model training for the remaining communication rounds. It is possible to see servers's and client's outputs below. In the example the client 1 will join the training after all the others clients joined:

```
# SERVER'S OUTPUT

Global Iteration: 29
Total number of clients: 4
Broadcasting global model
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Client connected: {'id': '1', 'port_no': 6001, 'data_size': 4150, 'model_sent': 0}
Global model average accuracy: 94.02%
Local average training loss:v 0.00662
Global Iteration: 30
Total number of clients: 5
Broadcasting global model
Getting local model from client: 2
Getting local model from client: 3
Getting local model from client: 4
Getting local model from client: 5
Getting local model from client: 1
Global model average accuracy: 81.81%
Local average training loss:v 0.01046


# CLIENT 1 OUTPUT

Sending handshake
I am client 1
Receiving new global model
Global Communication round: 30
Global Average Accuracy 94.02%
Global Average Training Loss 0.00662
Global model accuracy tested on local data: 32.80%
Local training...
Local training loss: 0.01343
Local model testing accuracy: 96.39%
Sending back new global model
```

As you can see the client joins the network between communication round 29 and 30, and this is correctly handled both by server (which broadcasts the model to all the 5 clients) and by the client (which knows the communication round and receives the current global model). This is achieved thanks to the ```HeartbeatThread``` which keeps listening for handshakes while the training is taking place.


## Analysis Results

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

Where the maximum accuracy has been reached with a batch size of 20 and it is equal to 92.48% at communication round 11.

<p align="center">
  <img src="https://user-images.githubusercontent.com/82953736/170230785-ceb35340-9df3-4a01-9c4a-8737e38da9cf.png" width="550px"/>
</p>


