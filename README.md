# Project.py

## Main
- Loads testing data using `load_data()`
- Gets a net trained on the `mnist` database using `get_net_from_file()`
- Prints a report using `net_report()`

## Functions
- `load_data()` loads training and testing data from a file
- `get_net_from_file()` loads a net from a `.npz` savefile
- `net_report()` generates a printable report of the net's performance

# Neural Network from scratch in Python

Simple python Q-Learning based Neural Network class that can be trained to do tasks such as recognizing digits using the mnist database
## Installation

Import NeuralNetwork

```py
from neuralnetwork import NeuralNetwork
```
    
## Usage/Examples

Get a net trained on the mnist database from a savefile and test it
```python
import numpy as np
from neuralnetwork import NeuralNetwork

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    training_data = list(zip(training_images, training_labels))
net = NeuralNetwork((1,1))
net.load_net()
print(f'{net.test(training_data):,}/10,000')
```
## Features
- Training using gradient descent
- Saving a net to a .npz file
- Loading a net from a .npz file
- Testing a net's performance
- Report of the net's performance and training setup
## Authors

- [@lorenzo-frittoli](https://github.com/lorenzo-frittoli)


## Demo

[![Demo](http://img.youtube.com/vi/YJYmyz6xzO0/0.jpg)](http://www.youtube.com/watch?v=YJYmyz6xzO0 "Lorenzo Frittoli CS50P Final")


## Documentation

[Documentation](https://github.com/lorenzo-frittoli/NeuralNetwork/blob/main/documentation.md)


## Optimizations
### Forward Propagation
#### Implementations
In the `forward_propagation()` method the net does not store the activation values. Instead, it uses a for loop in such a way that the output of the previous cycle is the input of the next one.

*An implementation similar to that in the class.*
```py
def fp_no_neurons(layer):
    for w,b in zip(weights, biases):
        layer = activate(w @ layer + b)
    return layer
```
*An implementation that stores neuron data.*
```py
def fp_neurons(activation):
    activations, weighted_sums = [], []
    for b, w in zip(biases, weights):
        weighted_sum = w @ activation + b
        weighted_sums.append(weighted_sum)
        activation = activate(weighted_sum)
        activations.append(activation)
```

I was forced to use a loop similar to the second function to generate the gradients in the `weights_biases_gradients()` function as activation data (as well as a couple other matrixes) are needed.

#### Performance
The first function is pretty consistently faster then the first one, taking on average 19% more.

*Tests with a net shape of (100,300,300,150), an input of `numpy.ones()` for 100,000 iterations, using the same weights and biases for both functions.*
| Without Neurons | With Neurons |
|--------|--------|
| 5.3 s  | 6.5 s  |
| 5.8 s  | 6.6 s  |
| 4.8 s  | 5.8 s  |

### Updating Weights and Biases
#### Implementations
In the `train_batch()` function it is needed to recalculate the weight and biases in function of their respective gradients.
I thought about two implementations, one using a single for loop and one using list comprehensions.

On small values the difference between the two functions is negligible.

*Implementation that uses one for loop.*
```py
def one_cicle():
    weights_, biases_ = [], []

    for w,b,wg,bg in zip(weights, biases, weights_gradient, biases_gradient):
        weights_.append(w-ratio*wg)
        biases_.append(b-ratio*bg)
    return weights_, biases_
```

*Implementation that uses two list comprehensions.*
```py
def list_comprehension():
    weights_ = [w-ratio*wg for w,wg in zip(weights,weights_gradient)]
    biases_ = [b-ratio*bg for b,bg in zip(biases,biases_gradient)]
    return weights_, biases_
```

#### Performance
The performance difference is pretty much negligible on 1,000 repetitions and, on higher repetitions, the results become quite inconsistent.

I have run a series of mesurements on 10,000 repetitions, but I did not include them here as they are too few to represent any valuable information given their inconsistency and I do not have the resources necessary to conduct extensive testing.

*Tests with a net shape of (100,300,300,150) an input of `numpy.ones()` for 1,000 iterations, using the same weights and biases for both functions.*
| For Loop | List Comprehension |
|----------|--------------------|
| 0.746 s  | 0.742 s            |
| 0.733 s  | 0.738 s            |
| 0.711 s  | 0.743 s            |

*Tests with a net shape of (100,300,300,150) an input of `numpy.ones()` for 5,000 iterations, using the same weights and biases for both functions.*

| For Loop | List Comprehension |
|----------|--------------------|
| 6.19 s   | 5.90 s             |
| 6.17 s   | 6.39 s             |
| 6.03 s   | 6.10 s             |

## Running Tests

To run tests, run the following command

```bash
  pytest test_project.py
```

## Lessons Learned

I learned how to make a proper github repo, with README, LICENSE and requirements files.

I learned how to implement the gradient descent algorithm and finally succeeded in making a Neural Network, something that I have wanted to do for years.

## License

[MIT](https://github.com/lorenzo-frittoli/NeuralNetwork/blob/main/LICENSE)

