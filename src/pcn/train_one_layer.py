from pcn.models import PCModel, PCTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn import optim
import os
from pcn import utils
import argparse
    
def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_train = torch.load('data/mnist_train.pt')
    dataset_valid = torch.load('data/mnist_valid.pt')
    dset_train = TensorDataset(dataset_train['images'][:cf.N], dataset_train['labels'][:cf.N])
    dset_valid = TensorDataset(dataset_valid['images'], dataset_valid['labels'])
    train_loader = DataLoader(
        dset_train, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )
    valid_loader  = DataLoader(
        dset_valid, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    
    if cf.schedule:
        optimizers = [
            optim.get_optim(
            model.layers,
            cf.optim,
            cf.lr,
            batch_scale=cf.batch_scale,
            grad_clip=cf.grad_clip,
            weight_decay=cf.weight_decay,
        )
        for l in range(model.n_layers)
        ]
        schedulers = [
            optim.ReduceLROnPlateau(optimizers[l], cf.factor, cf.patience, cf.threshold, cf.low_threshold, cf.min_lr) 
            for l in range(model.n_layers)
        ]
    else:
        optimizers = [
            optim.get_optim(
            model.layers,
            cf.optim,
            cf.lr,
            batch_scale=cf.batch_scale,
            grad_clip=cf.grad_clip,
            weight_decay=cf.weight_decay,
        )
        ]

    trainer = PCTrainer(model, optimizers)

    with torch.no_grad():
        for epoch in range(cf.n_epochs):

            train_errors = trainer.train(
                train_loader, epoch, cf.n_train_iters, cf.fixed_preds_train, cf.log
            )

            if cf.schedule:
                for l in range(model.n_layers):
                    metrics = train_errors[l+1]
                    schedulers[l].step(metrics)       

            if cf.schedule:
                if utils.early_stop(optimizers, cf.lr):
                    break 

    # Evaluate validation error
    valid_error = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)
    mode = 'a' if os.path.exists("outputs/one_layer_valid_error.txt") else 'w'
    with open("outputs/valid_error.txt", mode) as f:
        f.write(f"{cf.num_hidden}, {float(valid_error)} \n") # convert np.float64 to float for writing

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
    cf.n_epochs = 1000
    cf.log = False
    cf.factor = 0.5
    cf.threshold = 2e-4
    cf.low_threshold = 0.2
    cf.patience = 10

    # dataset params
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = True
    cf.batch_size = 64
    cf.N = 10097

    # optim params
    cf.schedule = True
    cf.optim = "Adam"
    cf.lr = 1e-4
    cf.min_lr = 1e-6
    cf.batch_scale = True
    cf.grad_clip = None
    cf.weight_decay = None

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.n_hidden = args.n_hidden
    cf.nodes = [cf.n_hidden, 784]
    cf.act_fn = utils.Tanh()

    main(cf)