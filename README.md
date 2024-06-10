# Neural Network Builder
![nn.jpeg](https://github.com/LoQiseaking69/neural-network-builder/blob/main/Nn.png)

Neural Network Builder is a PyQt5-based application that allows users to build, train, visualize, and save/load different types of neural networks. It supports feedforward, convolutional, and recurrent neural networks. The application provides a user-friendly interface for creating neural networks and managing their training and evaluation.
___
![img](https://github.com/LoQiseaking69/neural-network-builder/blob/main/NnB.jpg)
___
## Features

- **Build Neural Networks**: Create feedforward, convolutional, or recurrent neural networks with customizable layer sizes.
- **Train Neural Networks**: Train the created neural networks using custom datasets provided via CSV files or manual input.
- **Visualize Neural Networks**: Visualize the structure of feedforward neural networks and plot training loss over epochs.
- **Save and Load Models**: Save trained neural network models to disk and load them for later use.
- **TexP - Text Processing App**: A separate application for text processing, including data preprocessing, text analysis, visualization, and CSV organization
___
![texp](https://github.com/LoQiseaking69/neural-network-builder/blob/main/builder.png)
___https://github.com/LoQiseaking69/neural-network-builder/blob/main/builder.png
## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/LoQiseaking69/neural_network_builder.git
    cd neural_network_builder
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```sh
    python main.py
    ```

## Usage

### Building a Neural Network

1. **Select Model Type**: Choose between Feedforward, Convolutional, and Recurrent neural networks.
2. **Specify Layer Sizes**: Input the sizes for the input layer, hidden layer, and output layer.
3. **Build the Network**: Click on the "Build Neural Network" button to create the network.

### Training a Neural Network

1. **Upload CSV File**: Click the "Upload CSV File" button to upload a dataset. The CSV file should have input features in columns and the target values in the last column.
2. **Manual Input**: Alternatively, you can manually input the training data in the provided text fields.
3. **Train the Network**: Click on the "Train Neural Network" button to start training. The training loss will be displayed.

### Visualizing the Neural Network

1. **Visualize Structure**: For feedforward neural networks, click on the "Visualize Neural Network" button to see the network structure.
2. **Plot Training Loss**: Click on the "Plot Training Loss" button to visualize the training loss over epochs.

### Saving and Loading Models

1. **Save Model**: Click on the "Save Model" button to save the trained model to a file.
2. **Load Model**: Click on the "Load Model" button to load a previously saved model.

### Using TexP - Text Processing App

1. **Open TexP**: Click on the "TexP" button in the main application to open the TexP text processing app.
2. **Data Preprocessing**: Load and process text documents.
3. **Text Analysis**: Analyze the processed text data.
4. **Visualization**: Visualize the processed text data.
5. **CSV Organization**: Save the processed text data to a CSV file.

## Project Structure

```bash
neural_network_builder/
│
├── main.py
├── app/
│ ├── init.py
│ ├── app.py
│ └── styles.py
├── models/
│ ├── init.py
│ ├── feedforward.py
│ ├── convolutional.py
│ └── recurrent.py
└── utils/
├── init.py
├── visualization.py
└── texp_app.py
```
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bug fixes or enhancements.

## License

This project is licensed under the MIT License.
