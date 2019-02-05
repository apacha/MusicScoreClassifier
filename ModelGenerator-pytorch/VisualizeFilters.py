import os

import torch
from torch.nn import Module, Conv2d
import matplotlib.pyplot as plt


def visualize_filters():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading best model from check-point and testing...")
    checkpoint_directory = "checkpoints"
    best_model_name = os.listdir(checkpoint_directory)[0]
    best_model = torch.load(os.path.join(checkpoint_directory, best_model_name))  # type: Module
    best_model.cpu()

    model_layers = [i for i in best_model.children()]
    for model_layer in model_layers:
        if isinstance(model_layer, Conv2d):
            tensor = model_layer.weight.data.numpy()
            plot_kernels(tensor, 8)

def plot_kernels(tensor, number_of_columns):
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // number_of_columns
    fig = plt.figure(figsize=(number_of_columns, num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows, number_of_columns, i + 1)
        ax1.imshow(tensor[i][0,:,:], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    visualize_filters()
