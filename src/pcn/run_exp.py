import argparse
from pcn import utils, plotting
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pcn.models import PCModel
import os

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

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}.pt", map_location=utils.DEVICE, weights_only=True))

    # Create folder if it doesn't exist
    if not os.path.exists(f"outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}"):
        os.makedirs(f"outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}")

    # Weights as images
    fig, ax = plt.subplots()
    plotting.plot_samples(ax, model.layers[1].weights)
    fig.savefig(f"outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}/weights.png")

    # Replay
    img_batch, label_batch = next(iter(train_loader))

    # Get EC activities for episodes
    img_batch = utils.set_tensor(img_batch)
    with torch.no_grad():
        model.test_batch(
            img_batch, 
            n_iters=cf.n_max_iters, 
            step_tolerance=cf.step_tolerance, 
            init_std=cf.init_std, 
            fixed_preds=cf.fixed_preds_test
        )
    ec_batch = utils.set_tensor(model.mus[0])

    with torch.no_grad():
        model.replay_batch(
            ec_batch, 
            cf.n_max_iters,
            step_tolerance=cf.step_tolerance,
            init_std=cf.init_std,
            fixed_preds=cf.fixed_preds_test
        )

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].set_title('Observation')
    plotting.plot_samples(axes[0], img_batch, color=False)
    axes[1].set_title('Replay')
    plotting.plot_samples(axes[1], model.preds[-1], color=False)
    fig.savefig(f"outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}/replay.png")

    # Recall
    recall_size = 10    
    img_batch = img_batch[:recall_size]
    n_cut = img_batch.size(1)//2
    img_batch_half = utils.mask_image(img_batch, n_cut)
    img_batch_half = utils.set_tensor(img_batch_half)
    with torch.no_grad():
        model.recall_batch(
            img_batch_half, 
            cf.n_max_iters, 
            n_cut=n_cut, 
            step_tolerance=cf.step_tolerance,
            init_std=cf.init_std,
            fixed_preds=cf.fixed_preds_test
        )

    fig, axes = plt.subplots(recall_size, model.n_nodes, figsize=(5*model.n_nodes, 5*recall_size))
    for m in range(recall_size):
        for n in range(model.n_nodes):
            axes[m, n].plot(model.plot_batch_errors[m][n])
    fig.savefig(f"outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}/recall-convergence.png")

    fig, axes = plt.subplots(recall_size, 3, figsize = (5*3, 5*recall_size))
    axes[0, 0].set_title('Observation', fontsize=30)
    axes[0, 1].set_title('Recall', fontsize=30)
    axes[0, 2].set_title('Original', fontsize=30)
    for m in range(recall_size):
        plotting.plot_samples(axes[m, 0], img_batch_half[m], color=False)
        plotting.plot_samples(axes[m, 1], model.mus[-1][m], color=False)
        plotting.plot_samples(axes[m, 2], img_batch[m], color=False)
    fig.savefig(f'outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}/recall.png')

    dataset_test = torch.load('data/mnist_test.pt')
    dset_test = TensorDataset(dataset_test['images'], dataset_test['labels'])
    test_loader = DataLoader(
        dset_test, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )

    # Hierarchical representations of test data
    fig = plotting.plot_levels(model, test_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test)
    fig.savefig(f'outputs/pcn-N={cf.N}-n_ec={cf.n_ec}-schedule={cf.schedule}/latents.png')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the PC model trained on a dataset of size N"
    )
    parser.add_argument("--N", type=int, default=64, help="Enter training set size")
    parser.add_argument("--n_ec", type=int, default=2, help="Enter size of EC layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--schedule", type=bool, default=False, help="Enter scheduler use")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = True
    cf.batch_size = 64
    cf.N = args.N

    # optim params
    cf.schedule = args.schedule

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.n_max_iters = 10000
    cf.step_tolerance = 1e-5
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False    

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.n_ec = args.n_ec
    cf.nodes = [cf.n_ec, 35, 784]
    cf.act_fn = utils.Tanh()

    main(cf)