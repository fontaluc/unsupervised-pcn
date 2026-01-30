import logging
import os
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.utils import make_grid
import wandb
import torch
from IPython.display import Image, display, clear_output
from pcn import utils
import time
import tempfile

def plot_samples(ax, x, size=(28, 28)):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    if len(size) == 2: # grayscale
        x_grid = make_grid(x.view(-1, 1, size[0], size[1]), nrow=nrow).permute(1, 2, 0)
    else: # color
        x_grid = make_grid(x.view(-1, size[0], size[1], size[2]), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')

def plot_2d_latents(ax, z, y, markers='o'):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = np.array([palette[int(l)] for l in y])
    alphas = {'*': 1, 'o': 0.1}
    if len(np.unique(markers)) > 1:
        for marker in np.unique(markers):
            indices = np.where(np.array(markers) == marker)[0]
            ax.scatter(z[indices, 0], z[indices, 1], c=colors[indices], marker=marker, alpha = alphas[marker])
    else:
        ax.scatter(z[:, 0], z[:, 1], c=colors)
    ax.set_aspect('equal', 'box')

def plot_latents(ax, z, y, markers='o', tsne=True):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = np.array([palette[int(l)] for l in y])
    alphas = {'*': 1, 'o': 0.1}
    z = TSNE(n_components=2).fit_transform(z) if tsne else PCA(n_components=2).fit_transform(z)

    if len(np.unique(markers)) > 1:
        for marker in np.unique(markers):
            indices = np.where(np.array(markers) == marker)[0]
            ax.scatter(z[indices, 0], z[indices, 1], c=colors[indices], marker=marker, alpha = alphas[marker])
    else:
        ax.scatter(z[:, 0], z[:, 1], c=colors)

def visualize_latent(ax, z, y, markers='o', tsne=True):    
    try:
        if z.shape[1] == 2:
            ax.set_title('Latent Samples')
            plot_2d_latents(ax, z, y, markers)
        else:
            reduction = 't-SNE' if tsne else 'PCA'
            ax.set_title(f'Latent Samples ({reduction})')
            plot_latents(ax, z, y, markers, tsne)
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

def log_reconstruction(x, model, epoch, size=(28, 28)):

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
    plot_samples(ax, x, size)

    # plot the prediction
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Prediction')
    plot_samples(ax, x_pred, size)

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

def log_plots(model, x, y, epoch, size=(28, 28)):
    log_reconstruction(x, model, epoch, size)
    log_latents(model, y, epoch)

def make_pc_plots(model, x, y, training_errors, validation_errors, weights, size=(28, 28), tmp_img="tmp_pc_out.png"):
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
    plot_samples(axes[model.n_nodes, 0], x, size)
    axes[model.n_nodes, 1].set_title('Prediction')
    plot_samples(axes[model.n_nodes, 1], model.preds[-1], size)

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

def plot_levels(activities, labels, markers='o', tsne=True, vertical=True):    
    n_nodes = len(activities)
    if vertical:
        fig, axes = plt.subplots(n_nodes, 1, figsize=(5, 5*n_nodes), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, n_nodes, figsize=(5*n_nodes, 5), constrained_layout=True)
    for n in range(n_nodes):
        i = n_nodes - 1 - n if vertical else n
        axes[n].set_xlabel(f'Level {i}')
        y = torch.Tensor(labels)
        z = torch.Tensor(activities[i])
        visualize_latent(axes[n], z, y, markers, tsne)
    return fig

def infer_latents(model, dataloader, n_iters, step_tolerance, init_std, fixed_preds):
    activities = [[] for _ in range(model.n_nodes)]
    labels = []
    # Append neuron activities and labels of all batches
    with torch.no_grad():
        for x, y in dataloader:
            model.test_batch(
                x, n_iters, step_tolerance, init_std, fixed_preds=fixed_preds
            )            
            for n in range(model.n_nodes):
                activities[n] += model.mus[n].to('cpu').tolist()
            labels += y.tolist()
    return activities, labels

def visualize_samples(model, cf, activities_test, labels_test, ec_batch, labels, size=(28, 28)):
    activities = [[] for _ in range(model.n_nodes)]
    K = len(ec_batch)
    sample_size = K*cf.batch_size
    markers = ['*' for _ in range(sample_size)] + ['o' for _ in range(len(labels_test))]
    fig1, axes = plt.subplots(K//2, 2, figsize = (2*5, 5*K//2))
    for k in range(K):
        label = int(labels[k*cf.batch_size])
        with torch.no_grad():
            model.replay_batch(
                ec_batch[k],
                cf.n_max_iters,
                step_tolerance=cf.step_tolerance,
                init_std=cf.init_std,
                fixed_preds=cf.fixed_preds_test
            )
        
        i = k//2
        j = k%2
        axes[i, j].set_title(f'Class {label}')
        plot_samples(axes[i, j], model.preds[-1], size)
        
        for n in range(model.n_nodes -1):
            activities[n] += model.mus[n].to('cpu').tolist()
        activities[-1] += model.preds[-1].to('cpu').tolist()

    for n in range(model.n_nodes):
        activities[n] += activities_test[n]

    labels += labels_test
    
    fig2 = plot_levels(activities, labels, markers)
    return fig1, fig2