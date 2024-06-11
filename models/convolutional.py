import numpy as np

class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.conv_filter = np.random.randn(3, 3) * np.sqrt(2 / 9)
        self.fc_weights = np.random.randn(64, 10) * np.sqrt(2 / 64)
        self.fc_biases = np.zeros(10)

    def forward(self, inputs):
        self.conv_output = np.zeros((inputs.shape[0] - 2, inputs.shape[1] - 2))
        for i in range(inputs.shape[0] - 2):
            for j in range(inputs.shape[1] - 2):
                self.conv_output[i, j] = np.sum(inputs[i:i+3, j:j+3] * self.conv_filter)
        self.conv_output = np.maximum(0, self.conv_output)  # ReLU activation
        self.flattened = self.conv_output.flatten()
        self.fc_output = np.dot(self.flattened, self.fc_weights) + self.fc_biases
        return self.softmax(self.fc_output)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            for input_data, target in zip(inputs, targets):
                output = self.forward(input_data)
                error = target - output
                epoch_loss += np.sum(error**2)
                
                d_fc_weights = np.outer(self.flattened, error)
                d_fc_biases = error
                d_flattened = np.dot(self.fc_weights, error)
                d_conv_output = d_flattened.reshape(self.conv_output.shape)
                d_conv_output[self.conv_output <= 0] = 0  # Backprop through ReLU
                d_conv_filter = np.zeros_like(self.conv_filter)
                for i in range(input_data.shape[0] - 2):
                    for j in range(input_data.shape[1] - 2):
                        d_conv_filter += input_data[i:i+3, j:j+3] * d_conv_output[i, j]
                
                self.fc_weights += learning_rate * d_fc_weights
                self.fc_biases += learning_rate * d_fc_biases
                self.conv_filter += learning_rate * d_conv_filter
            
            loss_history.append(epoch_loss / len(inputs))
        
        self.loss_history = loss_history
        return loss_history

    def save_model(self, file_path):
        model_data = {'conv_filter': self.conv_filter, 'fc_weights': self.fc_weights, 'fc_biases': self.fc_biases}
        np.save(file_path, model_data)

    def load_model(self, file_path):
        model_data = np.load(file_path, allow_pickle=True).item()
        self.conv_filter = model_data['conv_filter']
        self.fc_weights = model_data['fc_weights']
        self.fc_biases = model_data['fc_biases']