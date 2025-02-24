import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

# Utility function to generate mini-batches from the dataset
def batch_generator(train_x, train_y, batch_size):
    """Shuffles the dataset and yields mini-batches."""
    perm = np.random.permutation(len(train_x))
    train_x, train_y = train_x[perm], train_y[perm]
    for i in range(0, len(train_x), batch_size):
        yield train_x[i:i + batch_size], train_y[i:i + batch_size]

# Abstract base class for activation functions
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

# Implementations of various activation functions
class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)

class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x):
        omega = np.exp(x) + 2
        delta = np.exp(2 * x) + np.exp(x) + 1
        return np.exp(x) * omega / delta

# Abstract base class for loss functions
class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

# Implementations of various loss functions
class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-9))

    def derivative(self, y_true, y_pred):
        return y_pred - y_true

# Class representing a layer in the MLP
class Layer:
    def __init__(self, fan_in, fan_out, activation_function, dropout_rate=0.0, weight_decay=1e-5):
        self.weights = np.random.uniform(-np.sqrt(6 / (fan_in + fan_out)), np.sqrt(6 / (fan_in + fan_out)), (fan_in, fan_out))
        self.biases = np.zeros((1, fan_out))
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

    def forward(self, h, training=True):
        self.input = h
        self.z = np.dot(h, self.weights) + self.biases
        self.output = self.activation_function.forward(self.z)

        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, self.output.shape) / (1 - self.dropout_rate)
            self.output *= self.dropout_mask

        return self.output

    def backward(self, h, delta):
        dz = delta * self.activation_function.derivative(self.z)
        self.delta_weights = np.dot(h.T, dz) + self.weight_decay * self.weights
        self.delta_biases = np.sum(dz, axis=0, keepdims=True)
        self.delta = np.dot(dz, self.weights.T)
        return self.delta_weights, self.delta_biases

# Class representing the Multilayer Perceptron
class MultilayerPerceptron:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, loss_grad, input_data):
        deltas_w, deltas_b = [], []
        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            input_to_layer = input_data if i == 0 else self.layers[i - 1].output
            delta_w, delta_b = layer.backward(input_to_layer, delta)
            deltas_w.append(delta_w)
            deltas_b.append(delta_b)
            delta = layer.delta
        return deltas_w[::-1], deltas_b[::-1]

    def train(self, train_x, train_y, val_x, val_y, loss_func, learning_rate=0.001, batch_size=64, epochs=100, rmsprop=False, beta=0.9):
        training_losses, validation_losses = [], []
        grad_squared = [np.zeros_like(layer.weights) for layer in self.layers] if rmsprop else None

        for epoch in range(epochs):
            batch_losses = []
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(batch_x)
                loss = loss_func.loss(batch_y, y_pred)
                batch_losses.append(loss)
                loss_grad = loss_func.derivative(batch_y, y_pred)
                dW, dB = self.backward(loss_grad, batch_x)

                for j, layer in enumerate(self.layers):
                    if rmsprop:
                        grad_squared[j] = beta * grad_squared[j] + (1 - beta) * (dW[j] ** 2)
                        layer.weights -= learning_rate * dW[j] / (np.sqrt(grad_squared[j]) + 1e-8)
                    else:
                        layer.weights -= learning_rate * dW[j]
                    layer.biases -= learning_rate * dB[j]

            training_losses.append(np.mean(batch_losses))
            val_loss = loss_func.loss(val_y, self.forward(val_x, training=False))
            validation_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {training_losses[-1]:.4f} - Val Loss: {val_loss:.4f}")

        return training_losses, validation_losses