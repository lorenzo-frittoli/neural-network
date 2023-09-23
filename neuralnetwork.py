import numpy as np
from math import ceil

class NeuralNetwork:
    def __init__(self, shape: tuple|list) -> None:
        """Simple class that implements a Q-Learning Neural Network

        Args:
            shape (tuple): network shape

        Raises:
            ValueError: wrong type for shape (tuple)
        """
        self.shape = shape
        self.learning_rate = 3
        
        # Initializes weights and biases with list comprehension
        self.weight_shapes = [(k,n) for n,k in zip(shape[:-1], shape[1:])]
        self.bias_shapes = [(n,1) for n in shape[1:]]
        
        # Initializes biases as zeros but weights on a gaussian to speed up the learning process
        self.weights = [np.random.standard_normal(s) for s in self.weight_shapes]
        self.biases = [np.zeros(s) for s in self.bias_shapes]
        
    
    def __str__(self) -> str:
        """Function called when displaying an instance of the class, e.g. print(net)

        Returns:
            str: string detailing the shape of the net
        """
        return f'Neural network of shape: {self.shape}'
    
    
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """Forward propagation that does not store neuron values

        Args:
            input_data (np.ndarray|list|tuple): input layer values

        Returns:
            np.ndarray: output layer
        """
        layer = input_data
        for w,b in zip(self.weights, self.biases):
            layer = self.activate(w@layer + b)
        return layer
    
    
    def train(self, training_data: list[tuple[np.ndarray, np.ndarray]], batch_size: int, repetitions: int) -> None:
        """Trains the neural network.

        Args:
            training_data (list[tuple[np.ndarray, np.ndarray]]): training data in the format [(input, label)]
            batch_size (int): size of batches. The net will train on a batch before updating itself
            repetitions (int): number of times the net should train on the data
        """
        print('Training...')
        
        for i in range(repetitions):
            print(f'Repetition {i}')
        
            for n in range(ceil(len(training_data) / batch_size)):
                batch = training_data[batch_size*n : batch_size*(n+1)]
                self.train_batch(batch)
                
    
    def train_batch(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Runs gradient descent on a batch of training data before applying
        the average of the results to the net's weights and biases

        Args:
            batch (list[tuple[np.ndarray, np.ndarray]]): batch of training data
        """
        
        # Initializes the gradients as zeros, because the program will add the gradient of each case to the total
        total_weights_gradient = [np.zeros(s) for s in self.weight_shapes]
        total_biases_gradient = [np.zeros(s) for s in self.bias_shapes]
        ratio = self.learning_rate / len(batch) # Calculated here to save computation time
        
        for input_data, desired_output in batch:
            weights_gradient, biases_gradient = self.weights_biases_gradients(input_data, desired_output)   # Gets the case's gradients
            
            # Adds the case's gradients to the totals
            total_biases_gradient = [t+g for t,g in zip(total_biases_gradient, biases_gradient)]
            total_weights_gradient = [t+g for t,g in zip(total_weights_gradient, weights_gradient)]
            
        # Applies the average of the gradients times the learning rate (ratio = lr/len(batch)) to the net's weights and biases
        self.weights = [w - g*ratio for w,g in zip(self.weights, total_weights_gradient)]
        self.biases = [b - g*ratio for b, g in zip(self.biases, total_biases_gradient)]
            
            
    def weights_biases_gradients(self, input_data: np.ndarray, desired_output: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generates the gradients for weights and biases. They are in the same function to have a single loop and thus save time

        Args:
            input_data (np.ndarray): values of the input layer
            desired_output (np.ndarray): expected values of the output layer

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: gradients of weights and biases respectively
        """
        
        # Initializes the arrays that store weighted sums, their activations and their derivatives
        # The activated sums array is initialized with the input data. This is incorrect, because the input data
        # has not been passed through the activation function, but it is needed to simplify the following loops.
        w_sums = []
        activated_w_sums = [input_data]
        derived_w_sums = []
        
        # This is a reimplementatioon of forward_propagation() that stores neuron data
        # This is because of performance, to see more check documentation
        for b, w in zip(self.biases, self.weights):
            w_sums.append(w @ activated_w_sums[-1] + b)
            activated_w_sums.append(self.activate(w_sums[-1]))
            derived_w_sums.append(self.activate_derived(w_sums[-1]))

        # Initializes the gradients for the last layer, as the weights' gradient formula is different for the last layer
        biases_gradient = [(activated_w_sums[-1] - desired_output) * derived_w_sums[-1]]
        weights_gradient = [biases_gradient[-1] @ activated_w_sums[-2].transpose()]
        
        # Iterates from the second-to-last layer up to the second, adding the gradients to the lists
        for l in range(2, len(self.shape)):
            l = -l
            biases_gradient.append(self.weights[l+1].transpose() @ biases_gradient[-1] * derived_w_sums[l])
            weights_gradient.append(biases_gradient[-1] @ activated_w_sums[l-1].transpose())
            
        # Because the previous iteration is backwards, the gradients need to be reversed
        return reversed(weights_gradient), reversed(biases_gradient)
    
    
    def save_net(self, filename:str ='net_data.npz') -> None:
        """Saves the net data to a npz file

        Args:
            filename (str, optional): file to save the data to. Defaults to 'net_data.npz'.
        """
        np.savez_compressed(filename, shape=np.array(self.shape), weights=np.array(self.weights, dtype=np.ndarray), biases=np.array(self.biases, dtype=np.ndarray))


    def load_net(self, filename: str='net_data.npz') -> None:
        """Loads a net from an npz file

        Args:
            filename (str, optional): File to load the net from. Defaults to 'net_data.npz'.
        """
        with np.load(filename, allow_pickle=True) as loaded_net:
            self.shape = tuple(loaded_net['shape'].tolist())
            self.weights = loaded_net['weights'].tolist()
            self.biases = loaded_net['biases'].tolist()


    def test(self, data: list[tuple[np.ndarray, np.ndarray]]) -> int:
        """Tests the net's performance over the data

        Args:
            data (list[tuple[np.ndarray, np.ndarray]]): data to evaluate the net on

        Returns:
            int: number of times the it makes a correct prediction
        """
        score = 0
        for input_data, label in data:
            prediction = self.forward_propagation(input_data)
            score += int(self.test_case(prediction, label)) # int(True) -> 1, int(False) -> 0
            
        return score
            
    
    @staticmethod
    def test_case(prediction: np.ndarray, label: np.ndarray) -> bool:
        """Evaluates if a prediction is correct or not based on the desired output

        Args:
            prediction (np.ndarray): output neuron's values
            label (np.ndarray): desired output

        Returns:
            bool: True if the prediction is correct, False if it's not
        """
        if np.argmax(prediction) == np.argmax(label):
            return True
        else:
            return False
        
    
    @staticmethod
    def activate(value: np.ndarray) -> np.ndarray:
        """Sigmoid activation function

        Args:
            value (np.ndarray): matrix to apply the function to

        Returns:
            np.ndarray: value passed through the sigmoid function
        """
        return 1 / (1+np.exp(-value))
    
    
    @staticmethod
    def activate_derived(value: np.ndarray) -> np.ndarray:
        """Derivative of the activation function

        Args:
            value (np.ndarray): matrix to apply the function to

        Returns:
            np.ndarray: value passed through the sigmoid function's derivative
        """
        # Calls the function from the class and not from self because it's a static method
        s = NeuralNetwork.activate(value)
        return s*(1-s)
    
    
    @property
    def learning_rate(self):
        return self._learning_rate
        
    
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate