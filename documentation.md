---
description: |
    API documentation for modules: neuralnetwork.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...



# Module `neuralnetwork` {#id}








## Classes



### Class `NeuralNetwork` {#id}




>     class NeuralNetwork(
>         shape: tuple | list
>     )


Simple class that implements a Q-Learning Neural Network


Args
-----=
**```shape```** :&ensp;<code>tuple</code>
:   network shape



Raises
-----=
<code>ValueError</code>
:   wrong type for shape (tuple)








#### Instance variables



##### Variable `learning_rate` {#id}








#### Static methods



##### `Method activate` {#id}




>     def activate(
>         value: numpy.ndarray
>     ) ‑> numpy.ndarray


Sigmoid activation function


Args
-----=
**```value```** :&ensp;<code>np.ndarray</code>
:   matrix to apply the function to



Returns
-----=
<code>np.ndarray</code>
:   value passed through the sigmoid function




##### `Method activate_derived` {#id}




>     def activate_derived(
>         value: numpy.ndarray
>     ) ‑> numpy.ndarray


Derivative of the activation function


Args
-----=
**```value```** :&ensp;<code>np.ndarray</code>
:   matrix to apply the function to



Returns
-----=
<code>np.ndarray</code>
:   value passed through the sigmoid function's derivative




##### `Method test_case` {#id}




>     def test_case(
>         prediction: numpy.ndarray,
>         label: numpy.ndarray
>     ) ‑> bool


Evaluates if a prediction is correct or not based on the desired output


Args
-----=
**```prediction```** :&ensp;<code>np.ndarray</code>
:   output neuron's values


**```label```** :&ensp;<code>np.ndarray</code>
:   desired output



Returns
-----=
<code>bool</code>
:   True if the prediction is correct, False if it's not





#### Methods



##### Method `forward_propagation` {#id}




>     def forward_propagation(
>         self,
>         input_data: numpy.ndarray
>     ) ‑> numpy.ndarray


Forward propagation that does not store neuron values


Args
-----=
input_data (np.ndarray|list|tuple): input layer values

Returns
-----=
<code>np.ndarray</code>
:   output layer




##### Method `load_net` {#id}




>     def load_net(
>         self,
>         filename: str = 'net_data.npz'
>     ) ‑> None


Loads a net from an npz file


Args
-----=
**```filename```** :&ensp;<code>str</code>, optional
:   File to load the net from. Defaults to 'net_data.npz'.




##### Method `save_net` {#id}




>     def save_net(
>         self,
>         filename: str = 'net_data.npz'
>     ) ‑> None


Saves the net data to a npz file


Args
-----=
**```filename```** :&ensp;<code>str</code>, optional
:   file to save the data to. Defaults to 'net_data.npz'.




##### Method `test` {#id}




>     def test(
>         self,
>         data: list[tuple[numpy.ndarray, numpy.ndarray]]
>     ) ‑> int


Tests the net's performance over the data


Args
-----=
**```data```** :&ensp;<code>list\[tuple\[np.ndarray, np.ndarray]]</code>
:   data to evaluate the net on



Returns
-----=
<code>int</code>
:   number of times the it makes a correct prediction




##### Method `train` {#id}




>     def train(
>         self,
>         training_data: list[tuple[numpy.ndarray, numpy.ndarray]],
>         batch_size: int,
>         repetitions: int
>     ) ‑> None


Trains the neural network.


Args
-----=
**```training_data```** :&ensp;<code>list\[tuple\[np.ndarray, np.ndarray]]</code>
:   training data in the format [(input, label)]


**```batch_size```** :&ensp;<code>int</code>
:   size of batches. The net will train on a batch before updating itself


**```repetitions```** :&ensp;<code>int</code>
:   number of times the net should train on the data




##### Method `train_batch` {#id}




>     def train_batch(
>         self,
>         batch: list[tuple[numpy.ndarray, numpy.ndarray]]
>     ) ‑> None


Runs gradient descent on a batch of training data before applying
the average of the results to the net's weights and biases


Args
-----=
**```batch```** :&ensp;<code>list\[tuple\[np.ndarray, np.ndarray]]</code>
:   batch of training data




##### Method `weights_biases_gradients` {#id}




>     def weights_biases_gradients(
>         self,
>         input_data: numpy.ndarray,
>         desired_output: numpy.ndarray
>     ) ‑> tuple[list[numpy.ndarray], list[numpy.ndarray]]


Generates the gradients for weights and biases. They are in the same function to have a single loop and thus save time


Args
-----=
**```input_data```** :&ensp;<code>np.ndarray</code>
:   values of the input layer


**```desired_output```** :&ensp;<code>np.ndarray</code>
:   expected values of the output layer



Returns
-----=
<code>tuple\[list\[np.ndarray], list\[np.ndarray]]</code>
:   gradients of weights and biases respectively




-----
Generated by *pdoc* 0.10.0 (<https://pdoc3.github.io>).