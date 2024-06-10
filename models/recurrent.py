import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.hidden_size = hidden_size
        self.wx = np.random.randn(input_size, hidden_size)
        self.wh = np.random.randn(hidden_size, hidden_size)
        self.wy = np.random.randn(hidden_size, output_size)
        self.bh = np.random.randn(hidden_size)
        self.by = np.random.randn(output_size)

    def forward(self, inputs):
        self.previous_hidden_state = np.zeros((inputs.shape[0], self.hidden_size))
        h = self.previous_hidden_state
        for t in range(inputs.shape[1]):
            h = np.tanh(np.dot(inputs[:, t], self.wx) + np.dot(h, self.wh) + self.bh)
        self.hidden_state = h
        y = np.dot(h, self.wy) + self.by
        return self.softmax(y)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for input_data, target in zip(inputs, targets):
                self.previous_hidden_state = np.zeros((input_data.shape[0], self.hidden_size))
                output = self.forward(input_data)
                error = target - output
                d_wy = np.dot(self.hidden_state.T, error)
                d_by = np.sum(error, axis=0)
                delta_h = np.dot(error, self.wy.T) * (1 - self.hidden_state**2)
                d_wh = np.dot(self.previous_hidden_state.T, delta_h)
                d_wx = np.dot(input_data.T, delta_h)
                d_bh = np.sum(delta_h, axis=0)
                self.wy += learning_rate * d_wy
                self.by += learning_rate * d_by
                self.wh += learning_rate * d_wh
                self.wx += learning_rate * d_wx
                self.bh += learning_rate * d_bh

    def save_model(self, file_path):
        model_data = {'wx': self.wx, 'wh': self.wh, 'wy': self.wy, 'bh': self.bh, 'by': self.by}
        np.save(file_path, model_data)

    def load_model(self, file_path):
        model_data = np.load(file_path, allow_pickle=True).item()
        self.wx = model_data['wx']
        self.wh = model_data['wh']
        self.wy = model_data['wy']
        self.bh = model_data['bh']
        self.by = model_data['by']
