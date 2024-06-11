import matplotlib.pyplot as plt
import numpy as np

def visualize_network(nn):
    if isinstance(nn, FeedforwardNeuralNetwork):
        visualize_feedforward_network(nn)
    elif isinstance(nn, ConvolutionalNeuralNetwork):
        visualize_convolutional_network(nn)
    elif isinstance(nn, RecurrentNeuralNetwork):
        visualize_recurrent_network(nn)
    else:
        raise ValueError("Unknown neural network type")

def visualize_feedforward_network(nn):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    layer_sizes = nn.layers
    max_layer_size = max(layer_sizes)
    v_spacing = 1.0 / float(max_layer_size)
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    # Draw neurons
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (max_layer_size - layer_size) / 2
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing, layer_top + m * v_spacing), v_spacing / 4,
                                color='skyblue', ec='black', zorder=4)
            ax.add_artist(circle)

    # Draw connections
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (max_layer_size - layer_size_a) / 2
        layer_top_b = v_spacing * (max_layer_size - layer_size_b) / 2
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing, (n + 1) * h_spacing],
                                  [layer_top_a + m * v_spacing, layer_top_b + o * v_spacing], c='grey', alpha=0.5)
                ax.add_artist(line)

    plt.title('Feedforward Neural Network Structure', fontsize=16)
    plt.show()

def visualize_convolutional_network(nn):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('Convolutional Neural Network Structure', fontsize=16)

    ax[0].set_title('Convolutional Layer')
    ax[0].imshow(nn.conv_filter, cmap='gray', interpolation='none')
    ax[0].axis('off')

    ax[1].set_title('Fully Connected Layer')
    ax[1].axis('off')

    # Draw neurons in the fully connected layer
    fc_size = nn.fc_weights.shape[1]
    for m in range(fc_size):
        circle = plt.Circle((0.5, (m + 1) / float(fc_size + 1)), 0.05, color='skyblue', ec='black', zorder=4)
        ax[1].add_artist(circle)

    plt.show()

def visualize_recurrent_network(nn):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    input_size = nn.input_size
    hidden_size = nn.hidden_size
    output_size = nn.output_size

    layer_sizes = [input_size, hidden_size, output_size]
    max_layer_size = max(layer_sizes)
    v_spacing = 1.0 / float(max_layer_size)
    h_spacing = 1.0 / 2  # Only two layers to show: input and recurrent block

    # Draw neurons for input, hidden and output layers
    for n, layer_size in enumerate([input_size, output_size]):
        layer_top = v_spacing * (max_layer_size - layer_size) / 2
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing, layer_top + m * v_spacing), v_spacing / 4,
                                color='skyblue', ec='black', zorder=4)
            ax.add_artist(circle)

    # Draw recurrent connections
    hidden_top = v_spacing * (max_layer_size - hidden_size) / 2
    for m in range(hidden_size):
        circle = plt.Circle((h_spacing, hidden_top + m * v_spacing), v_spacing / 4,
                            color='lightcoral', ec='black', zorder=4)
        ax.add_artist(circle)
        line = plt.Line2D([h_spacing, h_spacing], [hidden_top + m * v_spacing, hidden_top + (m + 1) * v_spacing], c='grey', alpha=0.5)
        ax.add_artist(line)

    plt.title('Recurrent Neural Network Structure', fontsize=16)
    plt.show()

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Over Time', fontsize=16)
    plt.grid(True)
    plt.show()