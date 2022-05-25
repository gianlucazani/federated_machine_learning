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

