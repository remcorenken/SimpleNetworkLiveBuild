# Here I will try to build the code for my own neural network
# I will build classes. The smallest class will be a node
# the class "above" will be a layer.
# the class above that will be a linked list.
# two special layer classes will be created that are the input and the output layer.


import numpy as np
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class Node:
    def __init__(self):
        self.value = 0
        self.offset = 0
        self.slope = 1
        self.activation_function = 'relu'
    def activation(self)->float:
        out_value = 0
        x = self.slope * self.value + self.offset
        match self.activation_function:
            case 'sigmoid':
                out_value = np.exp(x)/(1+np.exp(x))
            case 'relu':
                if x < 0:
                    out_value = 0
                else:
                    out_value = x
            case 'tanh':
                out_value = np.tanh(x)
            case _:
                raise ValueError('Invalid activation function')
        return out_value
class Layer:
    def __init__(self,number_nodes = 1, next_layer = None, prev_layer = None) -> None:
        self.list = [Node() for _ in range(2)]
        self.next = next_layer
        self.prev = prev_layer
        self.number = number_nodes
        self.mat = None # contains the weights of the input to this layer as x by y matrix where x is the number of nodes in the previous layer and y is the number of nodes in the current layer
    def __repr__(self):
        return f"Layer(number of nodes={self.number}))"
    def softmax(self):
        total = np.sum(node.value for node in self.list)
        for node in self.list:
            node.value=node.value/total


class Network:
    def __init__(self, layers_def=None) -> None:
        if layers_def is None:
            layers_def = [5, 2, 3] #default layers
        self.layer_list = [Layer(number_nodes=k, next_layer=None, prev_layer= None) for k in layers_def]
        for k in range(len(layers_def)):
            if k<len(layers_def)-1:
                self.layer_list[k].next = self.layer_list[k+1]
            if k>0:
                self.layer_list[k].prev = self.layer_list[k-1]
                x = self.layer_list[k-1].number # number of nodes in the prev layer will be number of rows in mat
                y = self.layer_list[k].number # number of columns in current row
                self.layer_list[k].mat = np.random.rand(x,y)
    def plt(self, cur_fig = None,draw_threshold: None|float = None) -> None:
        if cur_fig is None: #create a figure if needed.
            cur_fig = plt.figure()
        if draw_threshold is None:
            draw_threshold = -np.inf
        # clean the current axis
        cur_fig.clf()
        ax = cur_fig.add_subplot(1,1,1)
        # node location and plot nodes
        x=0
        cur_layer = self.layer_list[0]
        while True:
            x_now=x
            x_from=x-1
            for k in range(0,cur_layer.number):
                y_offset_now = -(cur_layer.number-1)/2
                ax.scatter(x_now,k+y_offset_now,color='blue')

                if cur_layer.prev is not None:
                    y_offset_prev = -(cur_layer.prev.number-1)/2
                    for l in range(0,cur_layer.prev.number):
                        w=cur_layer.mat[l,k]
                        if abs(w)>draw_threshold:
                            color = "red" if w > 0 else "blue"
                            ax.plot([x_from,x_now],[l+y_offset_prev,k+y_offset_now],color=color,linewidth= 3*w)
            if cur_layer.next is None:
                break
            x=x+1
            cur_layer=cur_layer.next



        plt.show(block=True)
        plt.pause(0.1)


def unity_test(test:str):
    plt.ion()
    match test:
        case 'network':
            tmp=Network()
            current=tmp.layer_list[0]
            while current:
                current = current.next
            tmp.plt(draw_threshold=None)
        case 'layer':
            tmp=Layer(number_nodes = 2, next_layer = None, prev_layer = None)
            print(tmp)
        case 'node':
            plt.subplots() # create the figure and axis
            plt.clf() #clear the figure
            for k in np.linspace(-5, 5, 20): #loop over a range of values
                node = Node()
                node.value = k
                node.offset = 0
                node.slope = 1
                node.activation_function = 'relu'
                plt.scatter(node.value, node.activation(),color='blue',marker='o',label='relu')
                node.activation_function = 'sigmoid'
                plt.scatter(node.value, node.activation(),color='red',marker='x',label='sigmoid')
                node.activation_function = 'tanh'
                plt.scatter(node.value, node.activation(),color='green',marker='s',label='tanh')

            # Get handles and labels
            handles, labels = plt.gca().get_legend_handles_labels()
            # Show only the first 3
            plt.legend(handles[:3], labels[:3])
            plt.show(block=False)
            plt.pause(0.1)
        case _:
            raise ValueError(f'Invalid test: {test}')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    unity_test('network')
    #unity_test('layer')
    #unity_test('node')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
