import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNet

def plot_losses(epochs: int, losses: np.ndarray):
    plt.plot(np.arange(1, epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses vs Epochs')
    plt.show()


def plot_report(neural_net: NeuralNet):
    epochs = neural_net.report["epochs"]
    epochs_list = np.arange(epochs)
    loss = neural_net.report["loss"]
    accuracy = neural_net.report["accuracy"]

    fig, ax1 = plt.subplots()

    ax1.plot(epochs_list, loss, 'steelblue', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='steelblue')
    ax1.tick_params('y', colors='steelblue')
    
    ax2 = ax1.twinx()

    ax2.plot(epochs_list, accuracy, 'firebrick', label='Accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Accuracy', color='firebrick')
    ax2.tick_params('y', colors='firebrick')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Neural Network Training Report')
    plt.show()