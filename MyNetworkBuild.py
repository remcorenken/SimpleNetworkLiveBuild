from Support import *





training_data=read_idx(os.path.join(get_valid_folder(), 'train-images.idx3-ubyte'))
training_labels=read_idx(os.path.join(get_valid_folder(),'train-labels.idx1-ubyte'))

nw = Network([28**2, 10]) #initialize the network
nw.set_input_layer(np.reshape(training_data[0],shape=(28**2,1)),255)# set the training sample scale to range
nw.forward_calculation()
nw.squeeze_function()
print(nw.layers[-1].node_values)

