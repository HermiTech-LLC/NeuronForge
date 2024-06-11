import numpy as np
import os
import requests

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
        omniverse_api_key = os.getenv('OMNIVERSE_API_KEY')
        if not omniverse_api_key:
            raise ValueError("Omniverse API key not found. Please set the API key.")

        # Ask for the Omniverse API URL
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
        
        # Send a POST request to the Omniverse API
        response = requests.post(api_url, json=data, headers={'Authorization': f'Bearer {omniverse_api_key}'})
        
        if response.status_code != 200:
            raise ValueError(f"Training failed: {response.text}")
        
        # Assume the API returns the training loss history
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