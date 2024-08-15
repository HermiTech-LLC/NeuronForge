# NeuronForge -Neural Network Builder
![nn.jpeg](https://github.com/LoQiseaking69/neural-network-builder/blob/main/Nn.png)

NeuronForge is a comprehensive PyQt5-based application designed for constructing, training, visualizing, and managing various types of neural networks. This tool supports feedforward, convolutional, and recurrent neural networks, seamlessly integrating with the Omniverse API to enhance the training process for recurrent models. With a user-friendly interface, this application simplifies the complex processes involved in neural network development and evaluation.
___
[![nn_builderVid](https://img.youtube.com/vi/GxW3TsPEaGA/0.jpg)](https://www.youtube.com/watch?v=GxW3TsPEaGA)
___

## Key Features

- **Neural Network Construction**: Effortlessly create feedforward, convolutional, or recurrent neural networks with customizable layer configurations.
- **Training Capabilities**: Train your neural networks using custom datasets from CSV files. Recurrent networks benefit from enhanced training capabilities through Omniverse API integration.
- **Visualization Tools**: Gain insights into your neural network structures and monitor training progress with detailed loss plots.
- **Model Management**: Save and load neural network models to/from disk, facilitating easy reuse and deployment.
- **TexP - Text Processing Application**: A dedicated tool for text data preprocessing, analysis, visualization, and CSV management.
- **API Integration**: Expose the functionality of the application via a FastAPI-based REST API for remote access and integration.
___
![texp](https://github.com/LoQiseaking69/neural-network-builder/blob/main/builder.png)
___
https://github.com/LoQiseaking69/TextProcessor

## Installation Guide

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/LoQiseaking69/NeuroForge.git
    cd nn-Builder
    ```

2. **Set Up a Virtual Environment** (recommended):
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Launch the Application**:
    ```sh
    python3 main.py
    ```


> ## Usage Instructions
>
> ### Constructing a Neural Network

> 1. **Select Model Type**: Choose from Feedforward, Convolutional, or Recurrent neural networks.
> 2. **Configure Layer Sizes**: Specify sizes for input, hidden, and output layers.
> 3. **Build the Network**: Click the "Build Neural Network" button to initialize the network.

 ### Training a Neural Network

 1. **Upload CSV File**: Use the "Upload CSV File" button to upload your dataset. Ensure the CSV file has input features in columns and target values in the last column.
 2. **Train the Network**: Click the "Train Neural Network" button to start the training process. The application will display the training loss as it progresses.

   - **Feedforward Networks**: Standard training using the specified layer structure.
   - **Convolutional Networks**: Ensure the input size forms a perfect square for proper reshaping.
   - **Recurrent Networks**: Requires an Omniverse API key for training. Enter the API key in the designated field before training.

### Visualizing the Neural Network

1. **View Structure**: Click "Visualize Neural Network" to display the network's architecture.
2. **Plot Training Loss**: Click "Plot Training Loss" to visualize the training loss over epochs.

### Managing Models

1. **Save Model**: Click "Save Model" to store the trained model on disk.
2. **Load Model**: Click "Load Model" to load a previously saved model from disk.

### Using TexP - Text Processing App

1. **Open TexP**: Click the "TexP" button in the main application to launch the TexP text processing tool.
2. **Preprocess Data**: Load and preprocess text documents.
3. **Analyze Text**: Conduct detailed analysis of the processed text data.
4. **Visualize Data**: Create visual representations of text data.
5. **Organize CSVs**: Save processed text data into CSV format for further use.

## Project Structure

```bash
NeuronForge/
│
├── main.py
├── api.py
├── app/
│ ├── __init__.py
│ ├── app.py
│ └── styles.py
├── models/
│ ├── __init__.py
│ ├── feedforward.py
│ ├── convolutional.py
│ └── recurrent.py
└── utils/
    ├── __init__.py
    ├── visualization.py
    └── texp_app.py
```
## Enhancements and Updates

### Training Methods

- **CSV-Based Training**: Simplified data input process by exclusively using CSV files for training datasets.
- **Omniverse API Integration**: Recurrent neural networks now utilize the Omniverse API for enhanced training, necessitating an API key.
- **Robust Error Handling**: Improved error management to address input size mismatches and missing API keys.

### Model Improvements

- **Feedforward Neural Network**:
    - Optimized weight initialization techniques.
    - Enhanced training with efficient backpropagation and gradient descent algorithms.
- **Convolutional Neural Network**:
    - Integrated He initialization for convolutional filters.
    - Refined training processes for more accurate gradient updates.
- **Recurrent Neural Network**:
    - Leveraged Omniverse API for superior training.
    - Streamlined training process with advanced gradient descent and error management.

### API Integration

- **Create and Train Models**: Expose endpoints to create and train neural network models via FastAPI.
- **Model Management**: Endpoints to save, load, and list models.
- **Visualization and Monitoring**: Endpoints to visualize network architecture and plot training loss.


## Contributing

We welcome contributions! Please open an issue or submit a pull request for any bug fixes or enhancements.

## License

This project is licensed under the MIT License.
