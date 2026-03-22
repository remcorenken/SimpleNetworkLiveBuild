import numpy as np
import struct
import matplotlib.pyplot as plt
import os
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
## definitions
def sigmoid(x,x0=0,sigma=1):
    val=x-x0
    return 1 / (1 + np.exp(-sigma*x))
def relu(x,x0=0,sigma=1):
    val=x-x0
    return np.maximum(0, sigma*x)
def tanh(x,x0=0,sigma=1):
    val=x-x0
    return np.tanh(sigma*x)
def scale(x,x0=0,sigma=1):
    val=x-x0
    return val/sigma


## support classes
class Node:
    def __init__(self) -> None:
        self.value = 0 # value of the node
        self.offset = 0 #offset for squeezing function
        self.sigma = 1 #sigma for squeezing function
        self.method = 'sigmoid'

    def squeeze_function(self) -> None:
        val = self.value
        match self.method:
            case 'sigmoid':
                out = sigmoid(val, x0=self.offset, sigma=self.sigma)
            case 'relu':
                out = relu(val, x0=self.offset, sigma=self.sigma)
            case 'tanh':
                out = tanh(val, x0=self.offset, sigma=self.sigma)
            case 'scale':
                out = scale(val, x0=self.offset, sigma=self.sigma)
            case _:
                out = val
        self.value = out

class Layer:
    def __init__(self) -> None:
        self.weights = np.array([])
        self._nodes = np.array([], dtype=Node)
    def __str__(self) -> str:
        return f"weights: {self.weights}\nnodes: {self._nodes}"

    @property
    def node_values(self):
        return np.array([node.value for node in self._nodes])

    @node_values.setter
    def node_values(self, values):
        # Ensure it's iterable
        if not hasattr(values, "__iter__"):
            raise TypeError("nodes must be set with an iterable of values")
        # Ensure lengths match
        if len(values) != len(self._nodes):
            raise ValueError("Length of values must match number of nodes")
        # Assign values to each Node
        for node, v in zip(self._nodes, values):
            node.value = v

    def activate_nodes(self, nodes_prior_layer: np.ndarray) -> None:
        if not self.weights.all(): #if no weights are given I am in the first layer
            print("no weights set")
            return
        dim=self.weights.shape
        dim2=nodes_prior_layer.shape
        if not dim[0]==dim2[1]:
            print("wrong number of elements in prior layer")
            print(dim)
            print(dim2)
            return
            #in future raise proper exception.
        #calculate nodes_prior_layer*self.weights
        self.node_values = nodes_prior_layer@self.weights
class Network:
    def __init__(self, nodes_per_layer=None) -> None:
        if nodes_per_layer is None:
            nodes_per_layer = [5, 3, 5]
        self.nodes_per_layer = nodes_per_layer
        self.layers = []

    def __str__(self) -> str:
        return f"Nodes per layer{self.nodes_per_layer}\n"

    @property
    def num_layers(self):
        return len(self.nodes_per_layer)

    def clear_build_network(self) -> None:
        self.layers = []
        for i in range(0, self.num_layers):
            if i == 0:
                self.layers[0].nodes = np.array(Layer())
                self.layers[0].nodes = np.array(np.zeros(self.nodes_per_layer[0]))
            else:
                self.layers.append(Layer())
                self.layers[i].nodes = np.array(np.zeros([1, self.n_nodes_per_layer[i]]))
                self.layers[i].weights = np.array(np.random.uniform(-1, 1, [n_nodes_per_layer[i - 1], n_nodes_per_layer[i]]))

## support functions
def get_valid_folder() -> str:
    path1 = r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST" #modify for your local system(s)
    path2 = r"C:\Users\remco\PycharmProjects\SimpleNetworkLiveBuild\TrainDataMNIST"

    # 1. Try first file
    if os.path.exists(path1):
        return path1

    # 2. Try second file
    if os.path.exists(path2):
        return path2


    # 3. If both fail, ask the user via GUI
    print("Neither file found. Please select a file.")

    # Hide the root Tk window
    root = Tk()
    root.withdraw()

    filename = askopenfilename(
        title="Select your data file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if filename:
        return filename
    else:
        raise FileNotFoundError("No file selected and default files not found.")

def read_idx(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        # Read the remaining data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

if __name__ == '__main__':# Usage:
    ## reading files
    train_images = read_idx(os.path.join(get_valid_folder(), 'train-images.idx3-ubyte'))
    train_labels = read_idx(os.path.join(get_valid_folder(), 'train-labels.idx1-ubyte'))

    print(train_images.shape)  # (60000, 28, 28)
    print(train_labels.shape)  # (60000,)

    plt.imshow(train_images[0], cmap='gray')
    plt.title(f"Label: {train_labels[0]}")
    plt.show(block=True)

