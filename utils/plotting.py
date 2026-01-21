import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, save_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label="Ground Truth")
    plt.plot(y_pred, label="Prediction")
    plt.xlabel("Sample Index")
    plt.ylabel("Target")
    plt.title("Sparse NN Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
