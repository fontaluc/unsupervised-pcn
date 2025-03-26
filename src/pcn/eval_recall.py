import argparse
from tqdm import tqdm
from pcn import utils, plotting
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pcn.models import PCModel

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_train = torch.load('data/mnist_train.pt')
    dset_train = TensorDataset(dataset_train['images'][:cf.N], dataset_train['labels'][:cf.N])
    train_loader = DataLoader(
        dset_train, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )

    with open(f"models/pcn-{cf.N}.pkl", "rb") as f:
        layers = pickle.load(f)
    f.close()

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.layers = layers

    # Qualitatively on ten images
    test_size = 10
    img_batch, label_batch = next(iter(train_loader))
    img_batch = img_batch[:test_size]
    n_cut = img_batch.size(2)/2
    img_batch_half = utils.mask_image(img_batch, n_cut)
    img_batch_half = utils.set_tensor(img_batch_half)
    model.forward_test_time(
        img_batch_half, 
        cf.n_max_iters, 
        n_cut=n_cut, 
        step_tolerance=cf.step_tolerance,
        init_std=cf.init_std,
        fixed_preds=cf.fixed_preds_test
    )

    fig, axes = plt.subplots(test_size, model.n_nodes, figsize=(5*model.n_nodes, 5*test_size))
    for m in range(test_size):
        for n in range(model.n_nodes):
            axes[m, n].plot(model.plot_batch_errors[m][n])
    fig.savefig(f"outputs/pcn-{cf.N}/recall-convergence.png")

    fig, axes = plt.subplots(test_size, 3, figsize = (5*3, 5*test_size))
    axes[0, 0].set_title('Observation', fontsize=30)
    axes[0, 1].set_title('Recall', fontsize=30)
    axes[0, 2].set_title('Original', fontsize=30)
    for m in range(test_size):
        plotting.plot_samples(axes[m, 0], img_batch_half[m], color=False)
        plotting.plot_samples(axes[m, 1], model.mus[-1][m], color=False)
        plotting.plot_samples(axes[m, 2], img_batch[m], color=False)
    fig.savefig(f'outputs/pcn-{cf.N}/recall.png')

    # Quantitatively on the whole training set: average RMSE between recalled and original images
    rmse = 0 
    for img_batch, label_batch in tqdm(train_loader):
        img_batch_half = utils.mask_image(img_batch, cf.n_cut)
        img_batch_half = utils.set_tensor(img_batch_half)
        model.forward_test_time(
            img_batch_half, 
            cf.n_max_iters, 
            n_cut=n_cut, 
            step_tolerance=cf.step_tolerance,
            init_std=cf.init_std,
            fixed_preds=cf.fixed_preds_test
        )
        rmse += torch.sum(utils.rmse(img_batch, model.mus[-1])).item()
    rmse = rmse/cf.N
    with open("outputs/recall_rmse.txt", "w") as f:
        f.write(f"{cf.N, rmse}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the PC model trained on a dataset of size N"
    )
    parser.add_argument("--N", type=int, default=64, help="Enter training set size")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    cf.N = args.N
    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.n_max_iters = 10000
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False
    cf.step_tolerance = 1-5

    main(cf)