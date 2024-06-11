import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias = np.zeros(self.layers[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, inputs):
        self.a = [inputs]
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.sigmoid(z))
        return self.a[-1]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, inputs, targets, learning_rate=0.1, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            output = self.forward(inputs)
            loss = np.mean((targets - output) ** 2)
            loss_history.append(loss)

            # Backpropagation
            deltas = [None] * len(self.weights)
            deltas[-1] = (targets - output) * self.sigmoid_derivative(output)

            for i in range(len(deltas) - 2, -1, -1):
                deltas[i] = deltas[i + 1].dot(self.weights[i + 1].T) * self.sigmoid_derivative(self.a[i + 1])

            # Gradient descent update
            for i in range(len(self.weights)):
                self.weights[i] += self.a[i].T.dot(deltas[i]) * learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0) * learning_rate

        self.loss_history = loss_history
        return loss_history

    def save_model(self, file_path):
        model_data = {'weights': self.weights, 'biases': self.biases}
        np.save(file_path, model_data)

    def load_model(self, file_path):
        model_data = np.load(file_path, allow_pickle=True).item()
        self.weights = model_data['weights']
        self.biases = model_data['biases']