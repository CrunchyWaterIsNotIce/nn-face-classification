import pandas as pd
import numpy as np
from nnet import nn

df_train = pd.read_csv("faces_vs_nonfaces_train_32by32.csv")

neural_network = nn([1024, 8, 1], 0.001, 10, df_train, "relu") # output is sigmoid
neural_network.learn()