from pcn.models import PCModel, PCTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from pcn import utils
import argparse
    
def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_valid = torch.load('data/mnist_valid.pt')
    dset_valid = TensorDataset(dataset_valid['images'], dataset_valid['labels'])
    valid_loader  = DataLoader(
        dset_valid, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )

    model_name = f"pcn-n_vc={cf.n_vc}"
    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    trainer = PCTrainer(model)

    # Evaluate validation error
    valid_error = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)
    mode = 'a' if os.path.exists("outputs/valid_error.txt") else 'w'
    with open("outputs/valid_error.txt", mode) as f:
        f.write(f"{cf.n_vc}, {valid_error} \n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains a PCN with one hidden layer."
    )
    parser.add_argument("--n_hidden", type=int, default=100, help="Enter size of hidden layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
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
    cf.nodes = [cf.n_vc, 784]

    main(cf)