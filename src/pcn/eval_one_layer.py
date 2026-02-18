from pcn.models import PCModel, PCTrainer
import torch
import os
from pcn import utils
import argparse
from pcn import datasets
import numpy as np
from filelock import FileLock
import pandas as pd

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    device = 'cpu' if cf.force_cpu else utils.DEVICE

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.test_size, cf.normalize, g)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size, utils.seed_worker, g, device)

    nodes = [cf.n_vc, np.prod(size)]
    
    model_name = f"{cf.dataset}-n_vc={cf.n_vc}" if cf.train_size == None else f"{cf.dataset}-train_size={cf.train_size}-n_vc={cf.n_vc}"
    if cf.positive:
        model_name += "-positive"    
    
    model = PCModel(
        nodes=nodes, 
        mu_dt=cf.mu_dt, 
        act_fn=cf.act_fn, 
        use_bias=cf.use_bias, 
        kaiming_init=cf.kaiming_init, 
        positive=cf.positive,
        device=device
    )    
    model.load_state_dict(torch.load(f"models/pcn-{model_name}.pt", map_location=device, weights_only=True))

    trainer = PCTrainer(model)

    # Evaluate validation error
    valid_error = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)
    # Lock file to prevent overwriting when multiple processes run
    with FileLock(f"eval_one_layer.csv.lock"):
        print('Lock acquired.')
        data = [cf.dataset, cf.positive, cf.n_vc, valid_error]
        if os.path.exists("outputs/eval_one_layer.csv"):
            df = pd.read_csv("outputs/eval_one_layer.csv")
            df.loc[len(df)] = data
        else:
            df = pd.DataFrame([data], columns=['Dataset', 'Positive', 'VC size', 'Validation error'])
        df.to_csv('outputs/eval_one_layer.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains a PCN with one hidden layer."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_vc", type=int, default=100, help="Enter size of hidden layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--act_fn", choices=['sigmoid', 'tanh', 'relu', 'linear'], default='sigmoid', help="Enter activation function")
    parser.add_argument("--positive", action='store_true', help="Enable non-negative states")
    parser.add_argument("--force_cpu", action='store_true', help="Use CPU even if GPU is available (to avoid CUDA out of memory)")

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
    cf.force_cpu = args.force_cpu

    main(cf)