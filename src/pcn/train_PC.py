import shutil
from pcn.models import PCModel, PCTrainer
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn import optim
import os
from pcn import utils
from pcn import plotting
import argparse
    
def main(cf):
    os.environ["WANDB__SERVICE_WAIT"] = "300" # sometimes wandb takes more than 30s (the default time limit) to start
    wandb.login()
    wandb.init(project="unsupervised-pcn", config=cf)
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
        nodes=cf.nodes, 
        mu_dt=cf.mu_dt, 
        act_fn=cf.act_fn, 
        use_bias=cf.use_bias, 
        kaiming_init=cf.kaiming_init, 
        use_decay=cf.decay,
        use_norm=cf.norm
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
                train_loader, epoch, cf.n_train_iters, cf.fixed_preds_train, cf.log_freq
            )
            for n in range(model.n_nodes):
                wandb.log({f'errors_{n}_train': train_errors[n], 'epoch': epoch})

            if cf.schedule:
                for l in range(model.n_layers):
                    metrics = train_errors[l+1]
                    if epoch > 0:                
                        better_ratio, low_ratio = utils.compute_ratios(metrics, schedulers[l])
                        wandb.log({f'better_ratio_{l}': better_ratio, 'epoch': epoch})
                        wandb.log({f'low_ratio_{l}': low_ratio, 'epoch': epoch})
                    schedulers[l].step(metrics)
                    wandb.log({f'scheduler_count_{l}': schedulers[l].num_bad_epochs, 'epoch': epoch})
                    wandb.log({f'lr_{l}': optimizers[l].lr, 'epoch': epoch})        

            img_batch, label_batch = next(iter(valid_loader))            
            valid_errors = trainer.eval(
                img_batch,cf.n_test_iters, cf.fixed_preds_test
            )
            for n in range(model.n_nodes):
                wandb.log({f'errors_{n}_valid': valid_errors[n], 'epoch': epoch})

            plotting.log_mnist_plots(model, img_batch, label_batch, epoch) 

            if cf.schedule:
                if utils.early_stop(optimizers, cf.lr):
                    break

    wandb.finish()

    # Remove local media directory
    path = os.path.join(location, 'media')
    shutil.rmtree(path)  

    # Create models folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save model parameters
    torch.save(model.state_dict(), f"models/pcn-n_vc={cf.n_vc}-n_ec={cf.n_ec}-schedule={cf.schedule}-decay={cf.decay}.pt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that trains the PC model on a training set of size N"
    )
    parser.add_argument("--n_epochs", type=int, default=1000, help="Enter number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Enter learning rate")
    parser.add_argument("--N", type=int, default=10097, help="Enter training set size")
    parser.add_argument("--n_vc", type=int, default=400, help="Enter size of VC layer")
    parser.add_argument("--n_ec", type=int, default=10, help="Enter size of EC layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    parser.add_argument("--schedule", type=bool, default=False, help="Enter scheduler use")
    parser.add_argument("--decay", type=bool, default=False, help="Enter decay use")
    parser.add_argument("--grad_clip", type=float, default=None, help="Enter grad clip value")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed
    cf.n_epochs = args.n_epochs
    cf.log_freq = 1000 # steps
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
    cf.N = args.N

    # optim params
    cf.schedule = args.schedule
    cf.optim = "Adam"
    cf.lr = args.lr
    cf.min_lr = 0
    cf.batch_scale = True
    cf.grad_clip = args.grad_clip
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
    cf.n_vc = args.n_vc
    cf.n_ec = args.n_ec
    cf.nodes = [cf.n_ec, cf.n_vc, 784]
    cf.act_fn = utils.Tanh()
    cf.decay = args.decay

    main(cf)