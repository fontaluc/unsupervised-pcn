import argparse
from tqdm import tqdm
from pcn import utils, plotting
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pcn.models import PCModel, PCTrainer
import os

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_train = torch.load('data/mnist_train.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'])
    train_loader = DataLoader(
        dset_train, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )
    N = len(dataset_train)

    model_name = f"pcn-n_ec={cf.n_ec}"
    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    # Create folder if it doesn't exist
    if not os.path.exists(f"outputs/{model_name}"):
        os.makedirs(f"outputs/{model_name}")

    ## Recall performance
    # Qualitatively on ten images
    test_size = 10
    img_batch, label_batch = next(iter(train_loader))
    img_batch = img_batch[:test_size]
    n_cut = img_batch.size(1)//2
    img_batch_half = utils.mask_image(img_batch, n_cut)
    img_batch_half = utils.set_tensor(img_batch_half)
    with torch.no_grad():
        model.recall_batch(
            img_batch_half, 
            cf.n_max_iters, 
            n_cut=n_cut, 
            step_tolerance=cf.step_tolerance,
            init_std=cf.init_std,
            fixed_preds=cf.fixed_preds_test
        )

    fig, axes = plt.subplots(test_size, model.n_nodes, figsize=(5*model.n_nodes, 5*test_size))
    for m in range(test_size):
        for n in range(model.n_nodes):
            axes[m, n].plot(model.plot_batch_errors[m][n])
    fig.savefig(f"outputs/{model_name}/recall-convergence.png")

    fig, axes = plt.subplots(test_size, 3, figsize = (5*3, 5*test_size))
    axes[0, 0].set_title('Observation', fontsize=30)
    axes[0, 1].set_title('Recall', fontsize=30)
    axes[0, 2].set_title('Original', fontsize=30)
    for m in range(test_size):
        plotting.plot_samples(axes[m, 0], img_batch_half[m], color=False)
        plotting.plot_samples(axes[m, 1], model.mus[-1][m], color=False)
        plotting.plot_samples(axes[m, 2], img_batch[m], color=False)
    fig.savefig(f'outputs/{model_name}/recall.png')

    # Quantitatively on the whole training set: average RMSE between recalled and original images
    mse = 0 
    with torch.no_grad():
        for img_batch, label_batch in tqdm(train_loader):
            img_batch_half = utils.mask_image(img_batch, n_cut)
            img_batch_half = utils.set_tensor(img_batch_half)
            model.recall_batch(
                img_batch_half, 
                cf.n_max_iters, 
                n_cut=n_cut, 
                step_tolerance=cf.step_tolerance,
                init_std=cf.init_std,
                fixed_preds=cf.fixed_preds_test
            )
            img_batch = utils.set_tensor(img_batch)
            mse += torch.sum(utils.mse(img_batch, model.mus[-1])).item()
    recall_mse = mse/N

    ## Generalization performance
    dataset_test = torch.load('data/mnist_test.pt')
    dset_test = TensorDataset(dataset_test['images'], dataset_test['labels'])
    test_loader = DataLoader(
        dset_test, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )
    trainer = PCTrainer(model)
    with torch.no_grad():
        test_mse = trainer.test(test_loader, cf.n_max_iters, cf.fixed_preds_test)
    mode = 'a' if os.path.exists("outputs/eval_PC.txt") else 'w'
    with open("outputs/eval_PC.txt", mode) as f:
        f.write(f"{cf.n_ec}, {recall_mse}, {float(test_mse)} \n") # convert np.float64 to float for writing

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the recall performance of a PC model"
    )
    parser.add_argument("--n_ec", type=int, default=30, help="Enter size of EC layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
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
    cf.use_bias = True
    cf.kaiming_init = False
    cf.n_ec = args.n_ec
    cf.nodes = [cf.n_ec, 400, 784]
    cf.act_fn = utils.Tanh()

    main(cf)