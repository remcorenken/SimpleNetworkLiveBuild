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
    return 1 / (1 + np.exp(-np.clip(sigma*x,-500,500))) #cliping value of input in sigmoid
def ddx_sigmoid(x,x0=0,sigma=1):
    return sigmoid(x,x0=x0,sigma=sigma)*(1-sigmoid(x,x0=x0,sigma=sigma))

def relu(x,x0=0,sigma=1):
    val=x-x0
    return np.maximum(0, sigma*x)
def ddx_relu(x,x0=0,sigma=1):
    return 0 if x<0 else sigma

def tanh(x,x0=0,sigma=1):
    val=x-x0
    return np.tanh(sigma*x)
def scale(x,x0=0,sigma=1):
    val=x-x0
    return val/sigma
def ddx_scale(x,x0=0,sigma=1):
    return 1/sigma


## support classes
class Node:
    def __init__(self) -> None:
        self.value = 0 # value of the node
        self.offset = 0 #offset for squeezing function
        self.sigma = 1 #sigma for squeezing function
        self.method = 'sigmoid'
    def __str__(self) -> str:
        return f"value: {self.value}, offset: {self.offset}, sigma: {self.sigma}"

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
        self._nodes: list[Node] = [] #not a numpy array, just a normal array.
    def __str__(self) -> str:
        return (f"weights: {self.weights.shape}\n"
                f"nodes: {np.array(self._nodes).shape}\n"
                f"nodes_type: {type(self._nodes)}\n"
                f"last node: {self._nodes[-1]}")
    @property
    def nodes(self):
        return self._nodes

    @property
    def node_values(self):
        return np.array([node.value for node in self._nodes]).T #transpose makes sure it is a row vector again

    @node_values.setter
    def node_values(self, values) -> None:
        # Ensure it's iterable
        if not hasattr(values, "__iter__"):
            raise TypeError("nodes must be set with an iterable of values")
        # Ensure lengths match
        if len(self._nodes)==0:
            self._nodes = [Node() for _ in range(len(values))]
        if len(values) == len(self._nodes): #now I can set
            for node, v in zip(self._nodes, values):   # Assign values to each Node
                node.value = v
        else:
            raise ValueError("Length of values must match number of nodes")

    def activate_nodes(self, nodes_values_prior_layer: np.ndarray) -> None:
        if not self.weights.all(): #if no weights are given I am in the first layer
            print("no weights set")
            return
        dim=self.weights.shape
        dim2=nodes_values_prior_layer.shape
        if not dim[0]==dim2[1]:
            print("wrong number of elements in prior layer")
            print(dim)
            print(dim2)
            return
            #in future raise proper exception.
        #calculate nodes_prior_layer*self.weights
        self.node_values = (nodes_values_prior_layer@self.weights).ravel()

    def squeeze_function(self) -> None:
        for node in self._nodes:
            node.squeeze_function()
    def softmax_function(self) -> None:
        # softmax on final layer
        m = np.sum(self.node_values)  # get maximum value for nodes
        self.node_values = self.node_values/m

class Network:
    def __init__(self, nodes_per_layer=None) -> None:
        if nodes_per_layer is None:
            nodes_per_layer = [5, 3, 5]
        self.nodes_per_layer = nodes_per_layer
        self.layers = []
        self.clear_build_network()

    def __str__(self) -> str:
        return f"Nodes per layer{self.nodes_per_layer}\n"

    @property
    def num_layers(self):
        return len(self.nodes_per_layer)

    def set_input_layer(self, input_layer: np.ndarray,max_val:float = 1) -> None:
        self.layers[0].node_values = input_layer/max_val

    def clear_build_network(self) -> None:
        self.layers = [Layer()]
        for i in range(0, self.num_layers):
            if i == 0:
                self.layers[0].node_values = np.array(np.zeros(self.nodes_per_layer[0]))
                #there are no input weights for the first layer
            else:
                self.layers.append(Layer())
                self.layers[i].node_values = np.array(np.zeros(self.nodes_per_layer[i]))
                self.layers[i].weights = np.array(np.random.uniform(-1, 1, [self.nodes_per_layer[i - 1], self.nodes_per_layer[i]]))

    def forward_calculation(self):
        for i in range(1, len(self.nodes_per_layer)):
            self.layers[i].activate_nodes(self.layers[i - 1].node_values)
    def squeeze_function(self) -> None:
        for i in range(1, len(self.nodes_per_layer)): #never squeeze the first layer; last layer uses different sort of squeeze
            self.layers[i].squeeze_function()

        self.layers[-1].softmax_function() #perform softmax on last layer


## helper functions
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
    # ## reading files
    # train_images = read_idx(os.path.join(get_valid_folder(), 'train-images.idx3-ubyte'))
    # train_labels = read_idx(os.path.join(get_valid_folder(), 'train-labels.idx1-ubyte'))
    #
    # print(train_images.shape)  # (60000, 28, 28)
    # print(train_labels.shape)  # (60000,)
    #
    # plt.imshow(train_images[0], cmap='gray')
    # plt.title(f"Label: {train_labels[0]}")
    # plt.show(block=True)

    # checking Node class
    # n = Node()
    # print(n)

    # checking Layer
    l=Layer()
    l.node_values = np.array([0, 1, 2, 3])
    print(l)
    print(l.node_values)

    nw=Network([3, 2, 3])
    print(nw.layers[2].node_values)

