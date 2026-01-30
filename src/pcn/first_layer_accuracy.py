import argparse
from tqdm import tqdm
from pcn import utils, plotting, datasets
import torch
from pcn.models import PCModel
import os
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    df = pd.read_csv("outputs/eval_two_layers.csv")

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.test_size, cf.normalize, g)

    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size, utils.seed_worker, g)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size, utils.seed_worker, g)

    if cf.dataset == 'mnist':
        cf.n_vc = 450
    elif cf.dataset == 'fmnist':
        cf.n_vc = 750
    else:
        cf.n_vc = 2000

    nodes = [cf.n_ec, cf.n_vc, np.prod(size)]
        
    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}-n_ec={cf.n_ec}"
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    # Classification accuracy
    activities_train, labels_train = plotting.infer_latents(
        model, train_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_train = activities_train[1]
    y_train = labels_train
    clf = LogisticRegression(random_state=cf.seed).fit(X_train, y_train)

    activities_valid, labels_valid = plotting.infer_latents(
        model, valid_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_valid = activities_valid[1]
    y_valid = labels_valid
    valid_acc = clf.score(X_valid, y_valid)
    
    idx = df.index[(df['Dataset'] == cf.dataset) & (df['EC size'] == cf.n_ec)]
    df.loc[idx, ['Validation accuracy 1']] = valid_acc
    df.to_csv('outputs/eval_two_layers.csv', index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the classification accuracy of the first layer of a 2-layer PCN."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_ec", type=int, default=30, help="Enter size of EC layer")
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
    cf.label_scale = None
    cf.normalize = True
    cf.batch_size = 64

    # inference params
    cf.mu_dt = 0.01
    cf.n_max_iters = 10000
    cf.step_tolerance = 1e-5
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.n_ec = args.n_ec
    cf.use_bias = True
    cf.kaiming_init = False
    cf.act_fn = utils.Tanh()

    main(cf)
