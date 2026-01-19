from pcn.models import PCModel, PCTrainer
import torch
from torch.utils.data import random_split
from pcn import optim
import os
from pcn import utils
from pcn import plotting
from pcn import datasets
import argparse
import wandb
import shutil
import numpy as np
    
def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    model_name = f"{cf.dataset}-n_vc={cf.n_vc}" if cf.train_size == None else f"{cf.dataset}-train_size={cf.train_size}-n_vc={cf.n_vc}"
    os.environ["WANDB__SERVICE_WAIT"] = "300" # sometimes wandb takes more than 30s (the default time limit) to start
    wandb.login()
    wandb.init(project="unsupervised-pcn", config=cf, name=model_name)
    location = wandb.run.dir

    train_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.normalize)
    
    test_size = len(test_dataset)
    train_size = len(train_dataset) - test_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, test_size])

    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size)

    nodes = [cf.n_vc, np.prod(size)]
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    
    if cf.scheduler:
        optimizer = optim.get_optim(
            model.layers,
            cf.optim,
            cf.lr,
            batch_scale=cf.batch_scale
        )        
        scheduler = optim.ExponentialLR(optimizer, cf.gamma)  

    else:
        optimizer = optim.get_optim(
            model.layers,
            cf.optim,
            cf.lr,
            batch_scale=cf.batch_scale,
            grad_clip=cf.grad_clip,
            weight_decay=cf.weight_decay
        )

    trainer = PCTrainer(model, optimizer)

    with torch.no_grad():
        for epoch in range(cf.n_epochs):

            train_errors = trainer.train(
                train_loader, epoch, cf.n_train_iters, cf.fixed_preds_train, log=cf.log
            )
            for n in range(model.n_nodes):
                wandb.log({f'errors_{n}_train': train_errors[n], 'epoch': epoch})

            if cf.scheduler:
                scheduler.step()
                wandb.log({f'lr': optimizer.lr, 'epoch': epoch}) 

            img_batch, label_batch = next(iter(valid_loader))            
            valid_errors = trainer.eval(
                img_batch, cf.n_test_iters, cf.fixed_preds_test
            )
            for n in range(model.n_nodes):
                wandb.log({f'errors_{n}_valid': valid_errors[n], 'epoch': epoch})

            plotting.log_plots(model, img_batch, label_batch, epoch, size)

    wandb.finish()

    # Remove local media directory
    path = os.path.join(location, 'media')
    shutil.rmtree(path) 

    # Create models folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save model parameters
    torch.save(model.state_dict(), f"models/pcn-{model_name}.pt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains a PCN with one hidden layer."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--train_size", type=int, default=None, help="Enter training set size")
    parser.add_argument("--n_vc", type=int, default=100, help="Enter size of hidden layer")
    parser.add_argument("--n_epochs", type=int, default=200, help="Enter number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--scheduler", action='store_true', help="Enable learning rate scheduler")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.n_epochs = args.n_epochs
    cf.log = False
    cf.gamma = 0.99

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = args.train_size
    cf.test_size = None
    cf.normalize = True
    cf.batch_size = 64

    # optim params
    cf.scheduler = args.scheduler
    cf.optim = "Adam"
    cf.lr = 1e-4
    cf.batch_scale = True
    cf.grad_clip = None
    cf.weight_decay = None

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.n_max_iters = 10000
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.n_vc = args.n_vc
    cf.act_fn = utils.Tanh()

    main(cf)