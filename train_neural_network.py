import pandas as pd
from nnet import nn

df_train = pd.read_csv("faces_vs_nonfaces_train_32by32.csv") # shape: (29812, 1025) ;-;
df_test = pd.read_csv("faces_vs_nonfaces_test_32by32.csv") # shape: (4045, 1025) ;-;

neural_network = nn([1024, 8, 1], 0.001, 10, df_train, "relu") # output is sigmoid
neural_network.learn()