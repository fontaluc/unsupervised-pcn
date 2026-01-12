import shutil
from pcn.models import PCModel, PCTrainer
import wandb
import torch
from torch.utils.data import random_split
from pcn import optim
import os
from pcn import utils
from pcn import plotting
from pcn import datasets 
import argparse
import numpy as np

    
def main(cf):

    model_name = f"{cf.dataset}-n_vc={cf.n_vc}-n_ec={cf.n_ec}"
    os.environ["WANDB__SERVICE_WAIT"] = "300" # sometimes wandb takes more than 30s (the default time limit) to start
    wandb.login()
    wandb.init(project="unsupervised-pcn", config=cf, name=model_name)
    location = wandb.run.dir

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
    train_dataset, validation_dataset = random_split(train_dataset, [cf.train_size, test_size])

    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    valid_loader = datasets.get_dataloader(validation_dataset, cf.batch_size)

    nodes = [cf.n_ec, cf.n_vc, np.prod(size)]
    model = PCModel(
        nodes=nodes,
        mu_dt=cf.mu_dt, 
        act_fn=cf.act_fn, 
        use_bias=cf.use_bias, 
        kaiming_init=cf.kaiming_init
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
                train_loader, epoch, cf.n_train_iters, cf.fixed_preds_train, cf.log_freq
            )
            for n in range(model.n_nodes):
                wandb.log({f'errors_{n}_train': train_errors[n], 'epoch': epoch})
            
            if cf.scheduler:
                scheduler.step()
                wandb.log({f'lr': optimizer.lr, 'epoch': epoch})                           

            img_batch, label_batch = next(iter(valid_loader))            
            valid_errors = trainer.eval(
                img_batch,cf.n_test_iters, cf.fixed_preds_test
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
        description="Script that trains the PC model"
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--train_size", type=int, default=None, help="Enter training set size")
    parser.add_argument("--n_epochs", type=int, default=4000, help="Enter number of epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Enter learning rate")
    parser.add_argument("--n_vc", type=int, default=450, help="Enter size of VC layer")
    parser.add_argument("--n_ec", type=int, default=30, help="Enter size of EC layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--scheduler", type=bool, default=True, help="Enter scheduler use")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.n_epochs = args.n_epochs
    cf.log_freq = 1000 # steps
    cf.gamma = 0.99

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = cf.train_size
    cf.test_size = None
    cf.normalize = True
    cf.batch_size = 64

    # optim params
    cf.scheduler = args.scheduler
    cf.optim = "Adam"
    cf.lr = args.lr
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
    cf.n_vc = args.n_vc
    cf.n_ec = args.n_ec
    cf.act_fn = utils.Tanh()
    cf.kaiming_init = False

    main(cf)