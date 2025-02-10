import pickle
from pcn.models import PCModel
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn import optim
import numpy as np
import os
import shutil
from pcn import utils
from pcn import plotting
import argparse

def train(train_loader, model, optimizer, epoch, n_train_iters, fixed_preds_train):
    training_epoch_errors = [[] for _ in range(model.n_nodes)]
    for batch_id, (img_batch, label_batch) in enumerate(train_loader):
        img_batch = utils.set_tensor(img_batch)
        model.train_batch(
            img_batch, n_train_iters, fixed_preds=fixed_preds_train
        )
        errors = model.get_errors()
        optimizer.step(
            curr_epoch=epoch,
            curr_batch=batch_id,
            n_batches=len(train_loader),
            batch_size=img_batch.size(0),
        )

        # gather data for the current batch
        for n in range(model.n_nodes):
            training_epoch_errors[n] += [errors[:, n].mean().item()]
            
    # gather data for the full epoch
    training_errors = []
    for n in range(model.n_nodes):
        error = np.mean(training_epoch_errors[n])
        wandb.log({f'errors_{n}_train': error, 'epoch': epoch})
        training_errors.append(error)

    return training_errors    
        

def eval(valid_loader, model, epoch, n_test_iters, fixed_preds_test):
    # Validation on a single batch
    img_batch, label_batch = next(iter(valid_loader))
    img_batch = utils.set_tensor(img_batch)
    model.test_batch(
        img_batch, n_test_iters, fixed_preds=fixed_preds_test
    )
    errors = model.get_errors()
    for n in range(model.n_nodes):
        error = errors[:, n].mean().item()
        wandb.log({f'errors_{n}_valid': error, 'epoch': epoch})

    return img_batch, label_batch
    
def main(cf):
    wandb.init(project="unsupervised-pcn")
    location = wandb.run.dir

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
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    stop = False
    epoch = 0
    with torch.no_grad():
        while not stop:
            # Training
            training_errors = train( 
                train_loader, 
                model, 
                optimizer, 
                epoch, 
                cf.n_train_iters, 
                cf.fixed_preds_train)

            for l in range(model.n_layers):
                weights = model.get_weights()[l]
                wandb.log({f'weights_{l}': weights, 'epoch': epoch})

            # Validation
            img_batch, label_batch = eval( 
                valid_loader, 
                model, 
                epoch,
                cf.n_test_iters, 
                cf.fixed_preds_test
            )

            plotting.log_mnist_plots(model, img_batch, label_batch, epoch)

            # Stopping criteria
            if epoch > 0:
                for n in range(1, model.n_nodes):
                    stop =  stop & (abs(old_training_errors[n] -  training_errors[n]) < cf.fun_tolerance*(1 + abs(old_training_errors[n])))
            old_training_errors = training_errors
            epoch += 1

    wandb.finish()
    # Remove local media directory
    path = os.path.join(location, 'media')
    shutil.rmtree(path)

    # Save model parameters
    with open(f"models/pc-{cf.N}-params.pkl", "wb") as f:
        pickle.dump(model.params, f)

             

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains the PC model on a training set of size N"
    )
    parser.add_argument("--N", type=int, default=64, help="Enter training set size")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.fun_tolerance = 1e-5

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