import numpy as np
from utils import softmax

class NeuralLM:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(1)

        self.W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size))

        self.lr = 0.05

    def forward(self, X):
        self.hidden = np.tanh(np.dot(X, self.W1))
        logits = np.dot(self.hidden, self.W2)
        self.out = softmax(logits)
        return self.out

    def backward(self, X, y):
        error = self.out - y

        dW2 = np.dot(self.hidden.T, error)

        hidden_error = np.dot(error, self.W2.T)
        d_hidden = hidden_error * (1 - self.hidden ** 2)

        dW1 = np.dot(X.T, d_hidden)

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2

    def save(self, path="models/weights.npz"):
        np.savez(path, W1=self.W1, W2=self.W2)

    def load(self, path="models/weights.npz"):
        data = np.load(path)
        self.W1 = data["W1"]
        self.W2 = data["W2"]