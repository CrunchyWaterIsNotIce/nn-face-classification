import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, numpy.typing as npt

def relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.maximum(0, x)

def relu_derivative(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (x > 0).astype(np.float32)

def sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * (1 - x)

class layer:
    def __init__(self, num_n_in: int, num_n_out: int, type_active):
        self.bias = np.zeros((num_n_out, 1), dtype=np.float32)
        self.weights = np.random.uniform(-0.1, 0.1, (num_n_out, num_n_in)).astype(np.float32)
        self.input = None
        self.output = None
        self.activation = type_active
        
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        z = self.bias + self.weights @ input
        if self.activation == "sigmoid":
            self.output = sigmoid(z)
        elif self.activation == "relu":
            self.output = relu(z)
        return self.output
    
    def backward(self, delta: npt.NDArray[np.float32], learning: np.float32) -> npt.NDArray[np.float32]:
        # partial derivs of loss to out
        if self.activation == "sigmoid":
            d_activation = sigmoid_derivative(self.output)
        elif self.activation == "relu":
            d_activation = relu_derivative(self.output)
        
        delta *= d_activation

        weight_gradient = delta @ self.input.T # transposed to (1, num_inputs)
        bias_gradient = delta

        self.weights -= learning * weight_gradient
        self.bias -= learning * bias_gradient
        
        return self.weights.T @ delta

        
class nn:
    '''
    self.layers has a format of:
    [num. of neutrons in nth layer...]
    -Where the first and last index elements are considered the input and output layers.
    '''
    def __init__(self, ls, learning: np.float32, ep: int, data: pd.DataFrame, type_active: str):
        self.dataset = data
        self.learning_rate = learning
        self.epochs = ep
        self.layers = []
        self.activation = type_active
        
        for lay_i in range(len(ls) - 1):
            self.layers.append(layer(ls[lay_i], ls[lay_i + 1], type_active))
        self.layers[-1].activation = "sigmoid"
    
    def learn(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for data_i in range(len(self.dataset)):
                input = self.dataset.iloc[data_i][:-1].values.reshape(-1, 1).astype(np.float32)
                target = self.dataset.iloc[data_i]['target']
                target = np.array([[target]], dtype=np.float32) # make it so that it can evauluate multiple 0101
                
                for layer in self.layers:
                    input = layer.forward(input)
                
                delta_output = input - target # MSE and BCE derivitive
                total_loss += np.sum(delta_output ** 2)
                
                for layer in reversed(self.layers):
                    delta_output = layer.backward(delta_output, self.learning_rate)
            
            print(f"Epoch: {epoch + 1}/{self.epochs}\nLoss: {total_loss:.4f}")

    def predict(self, trainset: pd.DataFrame):
        correct = 0
        for test_i in range(len(trainset)):
            input = trainset.iloc[test_i][:-1].values.reshape(-1, 1).astype(np.float32)
            target = trainset.iloc[test_i]['target']
            target = np.array([[target]], dtype=np.float32) # make it so that it can evauluate multiple 0101
            
            for layer in self.layers:
                input = layer.forward(input)
            
            prediction = int((input > 0.5).item())
            if prediction == int(target.item()):
                correct += 1
        print(f"Test Accuracy: {correct / len(trainset) * 100:.2f}%")