import shutil
from matplotlib import pyplot as plt
from pcn.models import PCModule, PCTrainer
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn import optim
import os
from pcn import utils
from pcn import plotting
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

    model = PCModule(cf.nodes, cf.mu_dt, cf.act_fn, cf.use_bias, cf.kaiming_init)
    
    optimizers = [
        optim.get_optim(
            [model.layers[l]], # must be iterable
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

    trainer = PCTrainer(model, optimizers)

    # Logging
    wandb.login()
    wandb.init(project="unsupervised-pcn", config=cf)
    location = wandb.run.dir

    # Create models folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    for epoch in range(cf.n_epochs):

        training_errors = trainer.train(
            train_loader, epoch, cf.n_train_iters, cf.fixed_preds_train, cf.log_freq
        )
        for n in range(model.n_nodes):
            wandb.log({f'errors_{n}_train': training_errors[n], 'epoch': epoch})        

        for l in range(model.n_layers):
            metrics = training_errors[l+1]
            if epoch > 0:                
                better_ratio, low_ratio = utils.compute_ratios(metrics, schedulers[l])
                wandb.log({f'better_ratio_{l}': better_ratio, 'epoch': epoch})
                wandb.log({f'low_ratio_{l}': low_ratio, 'epoch': epoch})
            schedulers[l].step(metrics)
            wandb.log({f'scheduler_count_{l}': schedulers[l].num_bad_epochs, 'epoch': epoch})
            wandb.log({f'lr_{l}': optimizers[l].lr, 'epoch': epoch})

        img_batch, label_batch, validation_errors = trainer.eval(
            valid_loader, cf.n_test_iters, cf.fixed_preds_test
        )
        for n in range(model.n_nodes):
            wandb.log({f'errors_{n}_valid': validation_errors[n], 'epoch': epoch})

        plotting.log_mnist_plots(model, img_batch, label_batch, epoch)
        
        if utils.early_stop(optimizers, cf.lr):
            break

    torch.save(model.state_dict(), f"models/pcn-{cf.N}.pt")

    wandb.finish()
    
    # Remove local media directory
    path = os.path.join(location, 'media')
    shutil.rmtree(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains the PC model on a training set of size N"
    )
    parser.add_argument("--n_epochs", required=True, type=int, help="Enter number of epochs")
    parser.add_argument("--N", type=int, default=64, help="Enter training set size")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.n_epochs = args.n_epochs
    cf.factor = 0.5
    cf.threshold = 1e-4
    cf.low_threshold = 0.2
    cf.patience = 10
    cf.log_freq = 1000 # steps

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
    cf.min_lr = cf.factor*cf.lr
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
    cf.step_tolerance = 1-5

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.nodes = [2, 35, 784]
    cf.act_fn = utils.Tanh()

    main(cf)