import matplotlib.pyplot as plt

def visualize_network(nn):
    fig, ax = plt.subplots()
    ax.axis('off')

    layer_sizes = nn.layers
    max_layer_size = max(layer_sizes)
    v_spacing = 1.0 / float(max_layer_size)
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (max_layer_size - layer_size) / 2
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing, layer_top + m * v_spacing), v_spacing / 4,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (max_layer_size - layer_size_a) / 2
        layer_top_b = v_spacing * (max_layer_size - layer_size_b) / 2
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing, (n + 1) * h_spacing],
                                  [layer_top_a + m * v_spacing, layer_top_b + o * v_spacing], c='k')
                ax.add_artist(line)

    plt.title('Neural Network Structure')
    plt.show()

def plot_training_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()
