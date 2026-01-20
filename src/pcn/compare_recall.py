import argparse
from pcn import utils, plotting, datasets
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from pcn.models import PCModel
import numpy as np
from tqdm import tqdm

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_names = ['fmnist', 'cifar10']
    n_ec_list = [50, 100, 150, 200, 250, 300]
    K = len(n_ec_list)

    for dataset in dataset_names:
        
        train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(dataset, cf.train_size, cf.normalize)
        valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size)

        n_vc = 750 if dataset == 'fmnist' else 2000

        # Compute indices corresponding to the bottom half of images in the flattened tensor
        if len(size) == 3:
            C, H, W = size
            indices = torch.tensor([i*H*W + j*H + k for i in range(C) for j in range(H//2, H) for k in range(W)])
        else:
            H, W = size
            indices = torch.arange(H*W//2, H*W)
        
        test_size = 5
        img_batch, label_batch = valid_loader[0]
        img_batch = img_batch[:test_size]
        img_batch_half = utils.mask_image(img_batch, indices)
        img_batch_half = utils.set_tensor(img_batch_half)

        plt.rcParams.update({'font.size': 7})
        fig, axes = plt.subplots(test_size, K+2, figsize = ((K+2), test_size), constrained_layout=True)
        axes[0, 0].set_title('Observation')
        axes[0, K+1].set_title('Original')
        for i in range(test_size):
            plotting.plot_samples(axes[i, 0], img_batch_half[i], size)
            plotting.plot_samples(axes[i, K+1], img_batch[i], size)
        
        for i in tqdm(range(K)):

            n_ec = n_ec_list[i]
            model_name = f"pcn-{dataset}-n_vc={n_vc}-n_ec={n_ec}"
            nodes = [n_ec, n_vc, np.prod(size)]
            model = PCModel(
                nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
            )
            model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))
            
            with torch.no_grad():
                model.recall_batch(
                    img_batch_half, 
                    cf.n_max_iters, 
                    indices, 
                    step_tolerance=cf.step_tolerance,
                    init_std=cf.init_std,
                    fixed_preds=cf.fixed_preds_test
                )
            
            axes[0, i+1].set_title(f'$n_2={n_ec}$')
            for j in range(test_size):
                plotting.plot_samples(axes[j, i+1], model.preds[-1][j], size)

        fig.savefig(f"outputs/compare_recall_{dataset}.png")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that compared the recall of PCNs with different number of EC neurons"
    )
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
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.n_max_iters = 20000
    cf.step_tolerance = 1e-5
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False    

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.n_vc = 450
    cf.act_fn = utils.Tanh()
    cf.decay = False

    main(cf)