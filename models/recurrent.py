import numpy as np
import os
import requests

class RecurrentNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.hidden_size = hidden_size
        self.wx = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.wh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.wy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

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

    def train(self, inputs, targets, learning_rate=0.001, epochs=100):
        omniverse_api_key = os.getenv('OMNIVERSE_API_KEY')
        if not omniverse_api_key:
            raise ValueError("Omniverse API key not found. Please set the API key.")

        api_url = input("Please enter the Omniverse API URL: ")
        if not api_url:
            raise ValueError("Omniverse API URL is required.")

        # Use the Omniverse API key in your training process
        data = {
            'inputs': inputs.tolist(),
            'targets': targets.tolist(),
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        response = requests.post(api_url, json=data, headers={'Authorization': f'Bearer {omniverse_api_key}'})
        
        if response.status_code != 200:
            raise ValueError(f"Training failed: {response.text}")
        
        loss_history = response.json().get('loss_history', [])
        return loss_history

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

    def apply_layer_normalization(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)

    def clip_gradients(self, gradients, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(grad**2) for grad in gradients))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            gradients = [grad * clip_coef for grad in gradients]
        return gradients

    def adam_optimizer(self, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, 'm'):
            self.m = [np.zeros_like(param) for param in gradients]
            self.v = [np.zeros_like(param) for param in gradients]
            self.t = 0
        self.t += 1
        self.m = [beta1 * m + (1 - beta1) * grad for m, grad in zip(self.m, gradients)]
        self.v = [beta2 * v + (1 - beta2) * (grad**2) for v, grad in zip(self.v, gradients)]
        m_hat = [m / (1 - beta1**self.t) for m in self.m]
        v_hat = [v / (1 - beta2**self.t) for v in self.v]
        updates = [param - learning_rate * m / (np.sqrt(v) + epsilon) for param, m, v in zip(gradients, m_hat, v_hat)]
        return updates