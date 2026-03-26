import numpy as np
import matplotlib.pyplot as plt

def plot_losses(epochs: int, losses: np.ndarray):
    plt.plot(np.arange(1, epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses vs Epochs')
    plt.show()
