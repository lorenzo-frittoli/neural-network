from neuralnetwork import NeuralNetwork
import numpy as np


def main():
    """Loads and tests net trained on the mnist database
    """
    _, test_data = load_data()
    net = get_net_from_file()
    score = net.test(test_data)
    print(net_report((784,10,10), 1.5, 10_000, 20, 50, score))
    
    
def load_data(filename='mnist.npz') -> tuple[np.ndarray, np.ndarray]:
    """Loads training and testing data from a file
    
    Args:
        filename (str, optional): Name of the data file. Defaults to 'mnist.npz'.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: training data, test data
    """
    print('Loading Data...')
    with np.load(filename) as data:
        training_images = data['training_images']
        training_labels = data['training_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
    training_data = list(zip(training_images, training_labels))
    test_data = list(zip(test_images, test_labels))
    print('Data Loaded')
    
    return training_data, test_data


def get_net_from_file(filename='net_data.npz') -> NeuralNetwork:
    """Loads neurons from a file in a net

    Args:
        filename (str, optional): Name of the neurons file. Defaults to 'net_data.npz'.

    Returns:
        NeuralNetwork: Net loaded with the neuron values
    """
    print('Loading Net...')
    NN = NeuralNetwork((1,1))
    NN.load_net(filename)
    print('Net Loaded')
    return NN


def net_report(net_shape:tuple[int], coefficient: float, data_size: int, batch_size: int, repetitions: int, score: int) -> str:
    """Generates a nice printable report of the net's performance and training details

    Args:
        score (int): cases correctly predicted by the net

    Returns:
        str: printable string detailing the net's performance
    """
    return f"""
TRAINING DETAILS:
    Network Shape:          {net_shape}
    Learning Coefficient:   {coefficient:.2f}
    Data Size:              {data_size:,}
    Batch Size:             {batch_size:,}
    Repetitions:            {repetitions:,}

RESULTS:
    Rate:                   {100*score/data_size:.2f} %
    Score:                  {score:,}/{data_size:,}
"""


if __name__ == '__main__':
    main()