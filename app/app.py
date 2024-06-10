from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QMessageBox, QLabel, QComboBox, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QTextEdit
from .styles import apply_styles
from models.feedforward import FeedforwardNeuralNetwork
from models.convolutional import ConvolutionalNeuralNetwork
from models.recurrent import RecurrentNeuralNetwork
from utils.visualization import visualize_network, plot_training_loss
from utils.texp_app import TexPApp  # Import TexPApp from utils folder
import numpy as np
import pandas as pd

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.nn = None
        self.inputs = None
        self.targets = None
        self.setWindowTitle("Neural Network Builder")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_tab(), "Build NN")
        self.tabs.addTab(self.train_tab(), "Train NN")
        self.tabs.addTab(self.visualize_tab(), "Visualize NN")
        self.tabs.addTab(self.model_tab(), "Save/Load Model")

        layout.addWidget(self.tabs)

        # Add TexP button
        texp_button = QPushButton("TexP")
        texp_button.clicked.connect(self.open_texp_app)
        layout.addWidget(texp_button)

        self.setLayout(layout)

        apply_styles(self)

    def open_texp_app(self):
        self.texp_window = TexPApp()
        self.texp_window.setGeometry(300, 200, 800, 600)  # Set the window to open smaller and centered
        self.texp_window.show()

    def build_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Feedforward", "Convolutional", "Recurrent"])
        model_type_layout.addWidget(self.model_type_combo)
        layout.addLayout(model_type_layout)

        layer_layout = QVBoxLayout()
        input_layer_layout = QHBoxLayout()
        input_layer_layout.addWidget(QLabel("Input Layer Size:"))
        self.input_layer_combo = QComboBox()
        self.input_layer_combo.addItems([str(i) for i in range(1, 101)])
        input_layer_layout.addWidget(self.input_layer_combo)
        layer_layout.addLayout(input_layer_layout)

        hidden_layer_layout = QHBoxLayout()
        hidden_layer_layout.addWidget(QLabel("Hidden Layer Size:"))
        self.hidden_layer_combo = QComboBox()
        self.hidden_layer_combo.addItems([str(i) for i in range(1, 101)])
        hidden_layer_layout.addWidget(self.hidden_layer_combo)
        layer_layout.addLayout(hidden_layer_layout)

        output_layer_layout = QHBoxLayout()
        output_layer_layout.addWidget(QLabel("Output Layer Size:"))
        self.output_layer_combo = QComboBox()
        self.output_layer_combo.addItems([str(i) for i in range(1, 101)])
        output_layer_layout.addWidget(self.output_layer_combo)
        layer_layout.addLayout(output_layer_layout)

        layout.addLayout(layer_layout)

        self.build_button = QPushButton("Build Neural Network")
        self.build_button.clicked.connect(self.build_nn)
        layout.addWidget(self.build_button)

        tab.setLayout(layout)
        return tab

    def train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input (comma-separated):"))
        self.input_entry = QLineEdit()
        input_layout.addWidget(self.input_entry)
        layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Expected Output (comma-separated):"))
        self.output_entry = QLineEdit()
        output_layout.addWidget(self.output_entry)
        layout.addLayout(output_layout)

        self.upload_button = QPushButton("Upload CSV File")
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        self.train_button = QPushButton("Train Neural Network")
        self.train_button.clicked.connect(self.train_nn)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        tab.setLayout(layout)
        return tab

    def visualize_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.visualize_button = QPushButton("Visualize Neural Network")
        self.visualize_button.clicked.connect(self.visualize_network)
        layout.addWidget(self.visualize_button)

        self.plot_loss_button = QPushButton("Plot Training Loss")
        self.plot_loss_button.clicked.connect(self.plot_training_loss)
        layout.addWidget(self.plot_loss_button)

        self.visualize_text = QTextEdit()
        self.visualize_text.setReadOnly(True)
        layout.addWidget(self.visualize_text)

        tab.setLayout(layout)
        return tab

    def model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_button)

        tab.setLayout(layout)
        return tab

    def build_nn(self):
        try:
            model_type = self.model_type_combo.currentText()
            input_size = int(self.input_layer_combo.currentText())
            hidden_size = int(self.hidden_layer_combo.currentText())
            output_size = int(self.output_layer_combo.currentText())
            layers = [input_size, hidden_size, output_size]

            if model_type == "Feedforward":
                self.nn = FeedforwardNeuralNetwork(layers)
            elif model_type == "Convolutional":
                self.nn = ConvolutionalNeuralNetwork()
            elif model_type == "Recurrent":
                self.nn = RecurrentNeuralNetwork(input_size, output_size)
            
            self.train_button.setEnabled(True)
            self.save_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"{model_type} Neural Network Built Successfully")
        except ValueError:
            QMessageBox.critical(self, "Error", "Please select valid sizes for the layers")

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.inputs = data.iloc[:, :-1].values
                self.targets = data.iloc[:, -1].values.reshape(-1, 1)
                self.train_button.setEnabled(True)
                QMessageBox.information(self, "Success", "File Uploaded Successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def train_nn(self):
        try:
            if hasattr(self, 'inputs') and hasattr(self, 'targets'):
                inputs = self.inputs
                targets = self.targets
            else:
                inputs = np.array([float(i) for i in self.input_entry.text().split(",")])
                targets = np.array([float(i) for i in self.output_entry.text().split(",")])
                model_type = self.model_type_combo.currentText()
            
                if model_type == "Convolutional":
                    input_dim = int(np.sqrt(inputs.size))
                    if input_dim * input_dim != inputs.size:
                        raise ValueError("Input size must be a perfect square for Convolutional Neural Network")
                    inputs = inputs.reshape((input_dim, input_dim))
                    targets = targets.reshape(1, -1)
                elif model_type == "Recurrent":
                    inputs = inputs.reshape(1, -1)
                    targets = targets.reshape(1, -1)
                else:
                    if inputs.shape[0] != int(self.input_layer_combo.currentText()) or targets.shape[0] != int(self.output_layer_combo.currentText()):
                        raise ValueError("Input/Output size mismatch")
                    inputs = inputs.reshape(1, -1)
                    targets = targets.reshape(1, -1)
            
            loss_history = self.nn.train(inputs, targets)
            plot_training_loss(loss_history)
            QMessageBox.information(self, "Success", "Neural Network Trained Successfully")
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    def visualize_network(self):
        if isinstance(self.nn, FeedforwardNeuralNetwork):
            visualize_network(self.nn)
        else:
            QMessageBox.warning(self, "Warning", "Visualization is only available for Feedforward Neural Network")

    def plot_training_loss(self, loss_history=None):
        if loss_history is None:
            loss_history = getattr(self.nn, 'loss_history', [])
        if loss_history:
            plot_training_loss(loss_history)
        else:
            QMessageBox.warning(self, "Warning", "No training loss to plot")

    def save_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "NumPy files (*.npy)", options=options)
        if file_path:
            self.nn.save_model(file_path)
            QMessageBox.information(self, "Success", "Model Saved Successfully")

    def load_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "NumPy files (*.npy)", options=options)
        if file_path:
            model_type = self.model_type_combo.currentText()
            if model_type == "Feedforward":
                self.nn = FeedforwardNeuralNetwork([1, 1, 1])  # Placeholder sizes, will be replaced by loaded model
            elif model_type == "Convolutional":
                self.nn = ConvolutionalNeuralNetwork()
            elif model_type == "Recurrent":
                self.nn = RecurrentNeuralNetwork(1, 1)  # Placeholder sizes, will be replaced by loaded model
            
            self.nn.load_model(file_path)
            self.train_button.setEnabled(True)
            self.save_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Model Loaded Successfully")
