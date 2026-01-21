import numpy as np
import matplotlib.pyplot as plt
import SupportFunctions as sf

class Layer:
    def __init__(self):
        self.nodes = np.array([])
        self.weights = np.array([])




# define the cost function
# define number of layers
# define number of nodes per layer
# define the final layer (must be 10 classes)
# define the input layer (must be 28^2 by 1)
# define weight matrix per layer
# define normalization function
def normalization_sigmoid(x,x0,b):
    x=x-x0
    return 1/(1+np.exp(-b*x))

## main program
#layer 0 will be the input layer, layer 1..N-1 will be the intermediate layers, layer N will be the output layer
n_nodes_per_layer=[28**2,10] # input layer, output layer, these have fixed values no intermediate layers at this point
# define the input layer
#layer_input= np.zeros((1,28**2))
layers = [Layer()]  # now layer[0] exists

## test normalization func
#define a way to update the weights given cost function
#plt.subplots() # create the figure and axis
#
#plt.clf() #clear the figure
#for k in np.linspace(-5, 5, 20): #loop over a range of values
#    #print(normalization_sigmoid(k,0,1))
#    plt.scatter(k,normalization_sigmoid(k,0,1), color='red', marker='x', label='sigmoid')
#plt.show()

# load the data
training_data=sf.read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-images.idx3-ubyte")
training_labels=sf.read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-labels.idx1-ubyte")

#push first image into input layer
layer_input = np.reshape(training_data[0],(28**2,1))
# print(layer_input.shape)
# for i in range(256,300):
#    print(layer_input[i])
#check if loading worked.
#print(training_data.shape)  # (60000, 28, 28)
#print(training_labels.shape)  # (60000,)

#plt.imshow(training_data[0], cmap='gray')
#plt.title(f"Label: {training_labels[0]}")
#plt.show()

# define layer(s)
for i in range(0,len(n_nodes_per_layer)):
    if i == 0:
        layers[0].nodes=np.array(layer_input)
    else:
        layers.append(Layer())
        layers[i].nodes = np.array(np.zeros([1 ,n_nodes_per_layer[i]]))
        layers[i].weights=np.array(np.random.uniform(-1,1,[n_nodes_per_layer[i-1],n_nodes_per_layer[i]]))
print(layers[1].weights.shape)