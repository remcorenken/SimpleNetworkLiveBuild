import numpy as np
import matplotlib.pyplot as plt
import SupportFunctions as sf
import os

def normalization_sigmoid(x,b=1,x0=0):
    x=x-x0
    return 1/(1+np.exp(-b*x))
def dnormalization_sigmoid(x,x0=0,b=1):
    x=x-x0
    E=np.exp(-b*x)
    return -b*E/(1+E)**2

def normalization_sigmoid_layer(L):
    #print(L.nodes.shape)
    for i in range(len(L.nodes)):
        L.nodes[i]=normalization_sigmoid(L.nodes[i],L.b[i],L.x0[i])
    
def normalize_last_layer(L):
    if L.nodes.size > 2:
        S=np.sum(L.nodes)
        L.nodes = L.nodes/S
    else:
        pass

def dnormalize_last_layer(L):
    if L.nodes.size > 2:
        S=np.sum(L.nodes)
        return 1/S
    else:
        return 1
    
def dzdw_func(PL):
    return(PL.nodes)
    
class Layer:
    def __init__(self):
        self.nodes = np.array([])
        self.weights = np.array([])
        self.b = np.array([])
        self.x0 = np.array([])

    def __str__(self):
        return f"nodes({self.nodes.shape})\nweights({self.weights.shape})\n"
    
    def activate_nodes(self,nodes_prior_layer):
        # calculate z-value for each node in this layer based on the nodes of the prior layer and the weights of this layer.
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
#       do not do squasification here, as this is done in the forward calculation function.
#        for i in range(0,len(self.nodes)):
#            self.nodes[i]=normalization_sigmoid(self.nodes[i])
    




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
            normalization_sigmoid_layer(layers[i]) #currently fixed values for b and x0, in future these will need to be trained as well. 
def update_layer_backward_calculation():
    pass
    

        
def get_error(expected,actual):
    return np.sum((actual-expected)**2)/len(actual)
def get_derror(expected,actual):
    return 2*(actual-expected)/len(actual)
def bck_calc(layers,expected):
    # run the backward calculation starting at the last layer. 
    lr = 0.01 #define a learning rate for the weight update.
    # start the loop over layers backwards.
    for i in range(len(layers)-1,0,-1):
        # get the desired change per node 
        da =-1*(layers[i].nodes - expected)# calculate dz
        if i==(len(layers)-1): #last layer has a deviating normalization function. 
            dzda=dnormalize_last_layer(layers[i])  
        else:
            dzda=dnormalization_sigmoid(layers[i].nodes) #currently fixed values for b and x0, in future these will need to be trained as well.     
        dz=dzda*da
        # backpropagate the dz to the previous layer, 
        # i.e. calculate the desired activation of the previous layer using the current weights. 
        print(layers[i].weights.shape)
        expected = dz@layers[i].weights.T

        # calculate the desired change in weights for this layer. 
        # i.e. the change in weights that would have resulted in the desired dz assuming that the activation of the previous layer was correct. 
        # dz= layers[i-1].nodes*dw #see activation function of a layer
        pl=dzdw_func(layers[i-1]) 
        dw=np.linalg.pinv(pl.T@pl)@pl.T@dz
        # update the weights of this layer.
        layers[i].weights=layers[i].weights+lr*dw
        # update b and x0 of the normalization function of this layer.
        if i==(len(layers)-1):
            pass #do not update b and x0 of the last layer, as this is a different normalization function.
        else:
            # dbda = dzda*a/b nice little trick to get the derivative of the normalization function with respect to b and x0.
            dbda = dzda*layers[i].nodes/layers[i].b
            db= dbda*da
            layers[i].b=layers[i].b-lr*db
            dx0da=-dzda #x0 behaves the same as x but in oposite direction 
            dx0=dx0da*da
            layers[i].x0=layers[i].x0-lr*dx0

    # end loop    
    pass
    
def backward_calculation(layers,expected):
    # run the backward calculation starting at the last layer going towards the first
    # get de/dw use chain rule !!
    # a(L)=sigma(z) #sigma is the normalization function
    # z(L)=(sum(a(L-1)*w)-x0)
    # de/dw = dz/dw*da/dz*de/da
    # addapt w
    lr = 0.01
    for i in range(len(layers)-1,0,-1):
        deda=get_derror(expected,layers[i].nodes)# de/da
        if i==(len(layers)-1): #last layer has a deviating normalization function. 
            dadz=dnormalize_last_layer(layers[i]) # da/dz
        else:
            dadz=dnormalization_sigmoid(layers[i].nodes) #currently fixed values for b and x0, in future these will need to be trained as well.    
        #if activation in a node of previous layer is large, changing the weights has a large effect. 
        #as the forward calculation is activation * weight, these values directly form the dz/dw value
        dzdw=dzdw_func(layers[i-1])# dz/dw 
        print(dzdw.shape)
        print(dadz.shape)
        print(deda.shape)
        #calculate expected for previous layer by projecting (feeding back) dz of current layer via weigths to previous layer
        expected=layers[i].weights.T*dadz*deda
        dedw=dzdw.T*dadz*deda
        print(dedw.shape)
        layers[i].weights=layers[i].weights-lr*dedw
        print("loop completed this time")
 # How to modify the weights of the middle layer?
        
# I wanted to visualise the network as an image
# no need to fully understand, but it is fun though
def layer_plot(layers) -> None:
    plt.figure()
    for q in range(0,len(layers)): #loop over the layers
        nnodes=layers[q].nodes.shape[1]
        y = np.linspace(0,1,nnodes)
        x = q/(len(layers)-1)*np.ones_like(y) #get x position of layer
        activation=layers[q].nodes.flatten()+1
        plt.scatter(x,y,s=np.abs(activation)*100, c=activation, cmap='viridis',alpha=0.75,edgecolors='none')
        if q == 0: #for first layer the
            continue
        nnodes_prev=layers[q-1].nodes.shape[1]
        y_prev = np.linspace(0,1,nnodes_prev)
        x_prev = (q-1)/(len(layers)-1)*np.ones_like(y_prev)
        idx = 0
        for xp, yp in zip(x_prev, y_prev):
            for xn, yn in zip(x, y):
                i, j = np.unravel_index(idx, layers[q].weights.shape)
                w = layers[q].weights[i,j]*1
                plt.plot([xp, xn], [yp, yn], 'k-', linewidth=np.abs(w),alpha=0.5)

    plt.colorbar(label="Value (used for size and color)")
    plt.show(block=True)


## main program
#layer 0 will be the input layer, layer 1..N-1 will be the intermediate layers, layer N will be the output layer
n_nodes_per_layer=[1,5,5,1] # input layer, output layer, these have fixed values no intermediate layers at this point
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

x=np.random.rand(100)*6 #define 100 random points to evaluate. 
training_label,training_data=sf.GenSinTrainSet(x,0)#us a std of 0 i.e. no noise on input
#x is input, data is input to function containing noise, label is (desired) output without noise
#push first image into input layer
layer_input = np.reshape(training_data[0],(1,1))


# define layer(s)
# that is, build the network
for i in range(0,len(n_nodes_per_layer)):
    if i == 0:
        layers[0].nodes = np.array(np.zeros([1, n_nodes_per_layer[0]]))
    else:
        layers.append(Layer())
        layers[i].nodes = np.array(np.zeros([1 ,n_nodes_per_layer[i]]))
        layers[i].weights=np.array(np.random.uniform(-1,1,[n_nodes_per_layer[i-1],n_nodes_per_layer[i]]))
        layers[i].b=np.array(np.ones([1,n_nodes_per_layer[i]]))
        layers[i].x0=np.array(np.zeros([1,n_nodes_per_layer[i]]))

 ### copy mistake:    tmp3=dzdw(layers[i-1])# dz/dw
 ## training part.
 # loop over training examples
 # select the input and the label

## start loop over training itterations
# select a training example at random from the list
# forward calculation
# plot the network as a graph for q sec. 
# backward calculation + update weights, b-s, and x0-s
trial_selector = 0
 # set the input layer
layers[0].nodes = np.array(np.reshape(training_data[trial_selector],layers[0].nodes.shape))
 # define the expected outcome.
expected = training_label[trial_selector]
 # forward calculation
forward_calculation(layers)
#layer_plot(layers)
layer_plot(layers)
 
 # single backward calculation
bck_calc(layers,expected)

#build a loop to train the network over multiple training examples #run this multiple times
#show the real data
#show the result of the network (i.e. model) 
#show the current network as a graph
#end loop. 
 
 
 
