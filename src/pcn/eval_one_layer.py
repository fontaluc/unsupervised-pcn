from pcn.models import PCModel, PCTrainer
import torch
import os
from pcn import utils
import argparse
from pcn import datasets
import numpy as np
    
def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.test_size, cf.normalize, g)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size, utils.seed_worker, g)

    nodes = [cf.n_vc, np.prod(size)]

    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}-positive={cf.positive}" if cf.train_size == None else f"pcn-{cf.dataset}-train_size={cf.train_size}-n_vc={cf.n_vc}-positive={cf.positive}"
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    trainer = PCTrainer(model)

    # Evaluate validation error
    valid_error = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)
    mode = 'a' if os.path.exists(f"outputs/valid_error_positive={cf.positive}.txt") else 'w'
    with open(f"outputs/valid_error_positive={cf.positive}.txt", mode) as f:
        f.write(f"{cf.dataset}, {cf.n_vc}, {valid_error} \n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains a PCN with one hidden layer."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_vc", type=int, default=100, help="Enter size of hidden layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--act_fn", choices=['sigmoid', 'tanh', 'relu', 'linear'], default='sigmoid', help="Enter activation function")
    parser.add_argument("--positive", action='store_true', help="Enable non-negative states")
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
    cf.act_fn = args.act_fn
    cf.n_vc = args.n_vc
    cf.positive = args.positive

    main(cf)