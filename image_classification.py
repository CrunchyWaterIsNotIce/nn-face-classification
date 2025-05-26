from nnet import nn

neural_network = nn([1, 2, 1], 0.001, 1, None, "relu") # output is sigmoid
print(neural_network.layers)