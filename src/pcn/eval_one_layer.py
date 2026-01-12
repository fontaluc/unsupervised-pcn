from pcn.models import PCModel, PCTrainer
import torch
from torch.utils.data import random_split
import os
from pcn import utils
import argparse
from pcn import datasets
import numpy as np
    
def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    if cf.dataset == 'mnist':
        train_dataset = datasets.MNIST(train=True, size=cf.train_size, normalize=cf.normalize)
        test_dataset = datasets.MNIST(train=False, size=cf.test_size, normalize=cf.normalize)
        size = (28, 28)
    elif cf.dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(train=True, size=cf.train_size, normalize=cf.normalize)
        test_dataset = datasets.FashionMNIST(train=False, size=cf.test_size, normalize=cf.normalize)
        size = (28, 28)
    else:
        train_dataset = datasets.CIFAR10(train=True, size=cf.train_size, normalize=cf.normalize)
        test_dataset = datasets.CIFAR10(train=False, size=cf.test_size, normalize=cf.normalize)
        size = (3, 32, 32)
    
    test_size = len(test_dataset)
    train_size = len(train_dataset) - test_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, test_size])

    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size)

    nodes = [cf.n_vc, np.prod(size)]

    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}"
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    trainer = PCTrainer(model)

    # Evaluate validation error
    valid_error = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)
    mode = 'a' if os.path.exists(f"outputs/valid_error_{cf.dataset}.txt") else 'w'
    with open(f"outputs/valid_error_{cf.dataset}.txt", mode) as f:
        f.write(f"{cf.n_vc}, {valid_error} \n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains a PCN with one hidden layer."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_hidden", type=int, default=100, help="Enter size of hidden layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = None
    cf.test_size = None
    cf.normalize = True
    cf.batch_size = 64

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_max_iters = 10000
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.act_fn = utils.Tanh()
    cf.n_vc = args.n_hidden

    main(cf)