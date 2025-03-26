import logging
import os
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import wandb
import torch
from IPython.display import Image, display, clear_output
from pcn import utils
import time
import tempfile

def plot_samples(ax, x, color=False):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    if not color:
        x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    else:
        x_grid = make_grid(torch.cat((x.view(-1, 2, 14, 14), torch.zeros(x.shape[0], 1, 14, 14)), dim = 1), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')

def plot_2d_latents(ax, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = [palette[int(l)] for l in y]
    ax.scatter(z[:, 0], z[:, 1], color=colors)
    ax.set_aspect('equal', 'box')

def plot_latents(ax, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = [palette[int(l)] for l in y]
    z = TSNE(n_components=2).fit_transform(z) # n_samples > 30 (perplexity)
    ax.scatter(z[:, 0], z[:, 1], color=colors)

def visualize_latent(ax, z, y):    
    try:
        if z.shape[1] == 2:
            ax.set_title('Latent Samples')
            plot_2d_latents(ax, z, y)
        else:
            ax.set_title('Latent Samples (t-SNE)')
            plot_latents(ax, z, y)
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

def log_reconstruction(x, model, epoch, color=False):

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_img = tmp_file.name  # Get unique file name

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100) # disable imshow logging (Clipping input data to the valid range for imshow with RGB data)

    fig = plt.figure(figsize = (10, 5))

    x = x.to('cpu')
    x_pred = model.preds[-1].to('cpu')

    # plot the observation
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Observation')
    plot_samples(ax, x, color)

    # plot the prediction
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Prediction')
    plot_samples(ax, x_pred, color)

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)    

    wandb.log({'reconstruction': wandb.Image(tmp_img), 'epoch': epoch})
    os.remove(tmp_img)

    logger.setLevel(old_level)
    
def log_latents(model, y, epoch):

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_img = tmp_file.name  # Get unique file name

    fig, axes = plt.subplots(model.n_nodes, 1, figsize = (5, 5*model.n_nodes))
    
    # plot the latent samples
    try:
        for n in range(model.n_nodes):
            axes[n].set_ylabel(f'Level {model.n_nodes - n}')
            visualize_latent(axes[n], model.mus[n].detach(), y)
        
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)

    wandb.log({'latents': wandb.Image(tmp_img), 'epoch': epoch})
    os.remove(tmp_img)

def log_mnist_plots(model, x, y, epoch):
    log_reconstruction(x, model, epoch)
    log_latents(model, y, epoch)

def make_pc_plots(model, x, y, training_errors, validation_errors, weights, color=False, tmp_img="tmp_pc_out.png"):
    fig, axes = plt.subplots(model.n_nodes + 1, 4, figsize=(5*4, 5*(model.n_nodes + 1)), squeeze=False)
    for n in range(model.n_nodes):
        axes[n, 0].set_ylabel(f'Level {n}')
        visualize_latent(axes[n, 0], model.mus[n].detach(), y)       

        ax = axes[n, 2]
        ax.set_title('Squared prediction error')
        ax.plot(training_errors[n], label="Training")
        ax.plot(validation_errors[n], label="Validation")

    # Do not visualize topmost predictions, which are 0
    for n in range(1, model.n_nodes):
        visualize_latent(axes[n, 1], model.preds[n].detach(), y)
    axes[0, 1].set_visible(False)

    for l in range(model.n_layers):
        ax = axes[l, 3]
        ax.set_title(f'Weights {l}')
        ax.plot(weights[l])
    # Hide empty subplots
    for i in range(model.n_layers, model.n_nodes + 1):
        axes[i, 3].set_visible(False)

    axes[model.n_nodes, 0].set_title('Observation')
    plot_samples(axes[model.n_nodes, 0], x, color)
    axes[model.n_nodes, 1].set_title('Prediction')
    plot_samples(axes[model.n_nodes, 1], model.preds[-1], color)

    ax = axes[model.n_nodes, 2]
    ax.set_title('Sum of squared prediction errors')
    n_epochs = len(training_errors[0])
    total_training_errors = torch.zeros(n_epochs)
    total_validation_errors = torch.zeros(n_epochs)
    training_errors = torch.Tensor(training_errors) # size (model.n_nodes, n_epochs)
    validation_errors = torch.Tensor(validation_errors)
    for n in range(model.n_nodes):
        total_training_errors += training_errors[n]
        total_validation_errors += validation_errors[n]
    ax.plot(total_training_errors, label="Training")
    ax.plot(total_validation_errors, label="Validation")
    ax.legend()

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    # An error due to removing the image could stop the training process
    time.sleep(0.5)  # Ensure file is released
    # Attempt to remove the temporary image file 
    try:
        os.remove(tmp_img)
    except PermissionError as e:
        print(f"Warning: Could not remove temporary file '{tmp_img}' due to: {e}")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while removing '{tmp_img}': {e}")

def plot_levels(model, dataloader, n_iters, step_tolerance, init_std, fixed_preds):
    activities = [[] for _ in range(model.n_nodes)]
    labels = []
    
    # Append neuron activities and labels of all batches
    with torch.no_grad():
        for x, y in dataloader:
            x = utils.set_tensor(x)
            model.test_batch(
                x, n_iters, step_tolerance, init_std, fixed_preds=fixed_preds
            )            
            for n in range(model.n_nodes):
                activities[n] += model.mus[n].to('cpu').tolist()
            labels += y.tolist()

    fig, axes = plt.subplots(model.n_nodes, 1, figsize=(5, 5*model.n_nodes), constrained_layout=True)
    for n in range(model.n_nodes):
        axes[n].set_xlabel(f'Level {model.n_nodes - n}')
        z = torch.Tensor(activities[n])
        y = torch.Tensor(labels)
        visualize_latent(axes[n], z, y)

    return fig