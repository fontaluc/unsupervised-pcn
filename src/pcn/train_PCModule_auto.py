import pickle
from pcn.models import PCModule_auto, PCTrainer
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from pcn import utils
from pcn import plotting
import argparse
    
def main(cf):

    # Seed for reproducibility
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

    model = PCModule_auto(
        nodes=cf.nodes,  
        act_fn=cf.act_fn, 
        use_bias=cf.use_bias, 
        kaiming_init=cf.kaiming_init
    )
    trainer = PCTrainer(
        model, 
        torch.optim.SGD, 
        torch.optim.Adam, 
        cf.mu_dt, 
        cf.lr)
    
    # Logging
    wandb.login()
    wandb.init(project="unsupervised-pcn", config=cf)
    wandb.watch(model, log='all') # log gradients and parameter values
    location = wandb.run.dir

    # Create models folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    n_epochs_stable = [[] for l in range(model.n_layers)]
    epoch = 0
    while sum(n_epochs_stable) <= cf.patience*model.n_layers:

        training_errors = trainer.train(
            train_loader, cf.n_train_iters, cf.fixed_preds_train
        )
        for n in range(model.n_nodes):
            wandb.log({f'errors_{n}_train': training_errors[n], 'epoch': epoch})

        img_batch, label_batch, validation_errors = trainer.eval(
            valid_loader, cf.n_test_iters, cf.fixed_preds_test
        )
        for n in range(model.n_nodes):
            wandb.log({f'errors_{n}_valid': validation_errors[n], 'epoch': epoch})

        plotting.log_mnist_plots(model, img_batch, label_batch, epoch)

        if epoch > 0:
            # Stopping criteria
            for l in range(model.n_layers):
                if training_errors[l+1] < cf.error_ceil  and n_epochs_stable[l] <= cf.patience:
                    if abs(old_training_errors[l+1] -  training_errors[l+1]) < cf.fun_tolerance*(1 + abs(old_training_errors[l+1])):
                        n_epochs_stable[l] += 1
                        if n_epochs_stable[l] > cf.patience:
                            for param in model.layers[l].parameters():
                                param.require_grad = False

        old_training_errors = training_errors          
        epoch += 1

    wandb.finish()

    # Save model parameters
    torch.save(model.state_dict(), f"{location}/pc-{cf.N}.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains each layer of a PC model until convergence on a training set of size N."
    )
    parser.add_argument("--N", type=int, default=10097, help="Enter training set size")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.fun_tolerance = 1e-3
    cf.patience = 10
    cf.error_ceil = 1

    # dataset params
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = True
    cf.batch_size = 64
    cf.N = args.N

    # optim params
    cf.optim = "Adam"
    cf.lr = 1e-4
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
    cf.nodes = [2, 35, 784]
    cf.act_fn = utils.Tanh()

    main(cf)