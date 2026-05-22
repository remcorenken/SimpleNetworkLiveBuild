import numpy as np
import matplotlib.pyplot as plt
import SupportFunctions as sf
import os

def normalization_sigmoid(x,x0=0,b=1):
    x=x-x0
    return 1/(1+np.exp(-b*x))
def dnormalization_sigmoid(x,x0=0,b=1):
    x=x-x0
    E=np.exp(-b*x)
    return -b*E/(1+E)**2

def normalization_sigmoid_layer(L):
    #print(L.nodes.shape)
    for i in range(len(L.nodes)):
        L.nodes[i]=normalization_sigmoid(L.nodes[i])
    
def normalize_last_layer(L):
    S=np.sum(L.nodes)
    L.nodes = L.nodes/S

def dnormalize_last_layer(L):
    S=np.sum(L.nodes)
    return 1/S
    
def dzdw(PL):
    return(PL.nodes)
    
class Layer:
    def __init__(self):
        self.nodes = np.array([])
        self.weights = np.array([])
    def __str__(self):
        return f"nodes({self.nodes.shape})\nweights({self.weights.shape})\n"
    
    def activate_nodes(self,nodes_prior_layer):
        #check if calculation can be done
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
        self.nodes = nodes_prior_layer@self.weights
        for i in range(0,len(self.nodes)):
            self.nodes[i]=normalization_sigmoid(self.nodes[i],0,1e-3)
    




# define the cost function
## define number of layers
## define number of nodes per layer
## define the final layer (must be 10 classes)
## define the input layer (must be 28^2 by 1)
## define weight matrix per layer
# define normalization function

def forward_calculation(layers):
    for i in range(1,len(n_nodes_per_layer)): 
        layers[i].activate_nodes(layers[i-1].nodes)
        #apply the normalization_sigmoid to each node in the current layer
        if i==len(n_nodes_per_layer)-1: #Last layer special treatment
            normalize_last_layer(layers[i])
        else:
            normalization_sigmoid_layer(layers[i])
            
def backward_calculation(layers):
    pass
def get_error(expected,actual):
    return np.sum((actual-expected)**2)/len(actual)
def get_derror(expected,actual):
    return 2*(actual-expected)/len(actual)
    
# I wanted to visualise the network as an image
# no need to fully understand, but it is fun though
def layer_plot(layers) -> None:
    plt.figure()
    for q in range(0,len(layers)): #loop over the layers
        nnodes=layers[q].nodes.shape[1]
        y = np.linspace(0,1,nnodes)
        x = q/(len(layers)-1)*np.ones_like(y) #get x position of layer
        activation=layers[q].nodes.flatten()
        plt.scatter(x,y,s=np.abs(activation), c=activation, cmap='viridis',alpha=0.75,edgecolors='none')
        if q == 0: #for first layer the
            continue
        nnodes_prev=layers[q-1].nodes.shape[1]
        y_prev = np.linspace(0,1,nnodes_prev)
        x_prev = (q-1)/(len(layers)-1)*np.ones_like(y_prev)
        idx = 0
        for xp, yp in zip(x_prev, y_prev):
            for xn, yn in zip(x, y):
                i, j = np.unravel_index(idx, layers[q].weights.shape)
                w = layers[q].weights[i,j]*1e-2
                plt.plot([xp, xn], [yp, yn], 'k-', linewidth=np.abs(w),alpha=0.5)

    plt.colorbar(label="Value (used for size and color)")
    plt.show(block=False)


## main program
#layer 0 will be the input layer, layer 1..N-1 will be the intermediate layers, layer N will be the output layer
n_nodes_per_layer=[28**2,2,10] # input layer, output layer, these have fixed values no intermediate layers at this point
# define the input layer
#layer_input= np.zeros((1,28**2))
layers = [Layer()]  # now layer[0] exists

## test normalization func
#define a way to update the weights given cost function
#plt.subplots() # create the figure and axis

#plt.clf() #clear the figure
#for k in np.linspace(-5, 5, 20): #loop over a range of values
#    #print(normalization_sigmoid(k,0,1))
#    plt.scatter(k,normalization_sigmoid(k,0,1), color='red', marker='x', label='sigmoid')
#plt.show()

# load the data
# training_data=sf.read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-images.idx3-ubyte")
# training_labels=sf.read_idx(r"C:\Users\RenkenRJ\PyCharmMiscProject\TrainDataMNIST\train-labels.idx1-ubyte")
training_data=sf.read_idx(os.path.join(sf.get_valid_folder(), 'train-images.idx3-ubyte'))
training_labels=sf.read_idx(os.path.join(sf.get_valid_folder(),'train-labels.idx1-ubyte'))
#push first image into input layer
layer_input = np.reshape(training_data[0],(1,28**2))/255 #scale values between 0 and 1
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
# that is, build the network
for i in range(0,len(n_nodes_per_layer)):
    if i == 0:
        layers[0].nodes = np.array(np.zeros([1, n_nodes_per_layer[0]]))
    else:
        layers.append(Layer())
        layers[i].nodes = np.array(np.zeros([1 ,n_nodes_per_layer[i]]))
        layers[i].weights=np.array(np.random.uniform(-1,1,[n_nodes_per_layer[i-1],n_nodes_per_layer[i]]))
 
 ## training part.
 # loop over training examples
 # select the input and the label
trial_selector = 0
 # set the input layer
layers[0].nodes = np.array(np.reshape(training_data[trial_selector],(1,28**2))/255)
 # define the expected outcome.
#print(type(training_labels[trial_selector])) 
expected = np.zeros([1,n_nodes_per_layer[-1]])
#print(expected)
expected[0,training_labels[trial_selector]] = 1
 # forward calculation
forward_calculation(layers)
 # get error
err = get_error(expected,layers[-1].nodes)
# print('error')
# print(err)
 # get de/dw use chain rule !!
 # a(L)=sigma(z) #sigma is the normalization function
 # z(L)=(sum(a(L-1)*w)-x0)
 # de/dw = dz/dw*da/dz*de/da
 # addapt w
 #### currently x0 is fixed. In the future we can update this per node
 # also b is fixed, needs to be variabel and updatable per node
 # get de/dx0
 # de/dx0=dz/dx0*da/dz*de/da
 # addapt x0
 # more too do.
 ####
layer_plot(layers) 
## for last layer 
# print(expected)
tmp=get_derror(expected,layers[-1].nodes)# de/da
# print(tmp.shape)
tmp2=dnormalize_last_layer(layers[-1]) # da/dz
# print('de/dz')
# print((tmp2*tmp).shape)
tmp3=dzdw(layers[-2])# dz/dw
# print('shape tmp3')
# print(tmp3.shape)
# print('size of w final layer')
# print(layers[-1].weights.shape)
# print(tmp3)
dedw=tmp3.T*tmp2*tmp
# print(dedw)
#update weights final layer
#define learning rate
print(layers[-1].weights)
lr=0.01
#update last layer weights
layers[-1].weights=layers[-1].weights+lr*dedw
print(layers[-1].weights)
# for n in layers[-1].nodes:
    # tmp4=dnormalization_sigmoid(n)
    # print(n,tmp4)
# print(layers[-1])

 
 
 
#forward calculation
#forward_calculation(layers)
#layer_plot(layers)
#print(layers[1].nodes)
#print(layers[2].nodes)

#normalize_last_layer(layers[-1])
#print(layers[1].nodes)
#print(np.sum(layers[1].nodes))
#layer_plot(layers)
#plt.ion()
#plt.show()

#print(layers[0].nodes.shape)
#print(layers[1].nodes)
#layers[1].activate_nodes(layers[0].nodes)
#print(layers[1].nodes)
#print(normalization_sigmoid(layers[1].nodes))