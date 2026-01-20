import argparse
from pcn import utils, plotting, datasets
import torch
import matplotlib.pyplot as plt
from pcn.models import PCModel
from tqdm import tqdm
import numpy as np

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    dataset_names = ['fmnist', 'cifar10']
    n_ec_list = [50, 100, 150, 200, 250, 300]
    K = len(n_ec_list) 
    

    for dataset in dataset_names:
        
        train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(dataset, cf.train_size, cf.normalize)
        train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
        img_batch, label_batch = train_loader[0]

        n_vc = 750 if dataset == 'fmnist' else 2000

        fig, axes = plt.subplots(1, K+1, figsize = (5*(K+1), 5))
        axes[0].set_title('Observation')
        plotting.plot_samples(axes[0], img_batch, size)

        for i in tqdm(range(K)):
            n_ec = n_ec_list[i]

            model_name = f"pcn-{dataset}-n_vc={n_vc}-n_ec={n_ec}"
            nodes = [n_ec, n_vc, np.prod(size)]

            model = PCModel(
                nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
            )
            model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

            # Get EC activities for episodes
            img_batch = utils.set_tensor(img_batch)
            with torch.no_grad():
                model.test_batch(
                    img_batch, 
                    n_iters=cf.n_max_iters, 
                    step_tolerance=cf.step_tolerance, 
                    init_std=cf.init_std, 
                    fixed_preds=cf.fixed_preds_test
                )
            ec_batch = utils.set_tensor(model.mus[0])

            with torch.no_grad():
                model.replay_batch(
                    ec_batch, 
                    cf.n_max_iters,
                    step_tolerance=cf.step_tolerance,
                    init_std=cf.init_std,
                    fixed_preds=cf.fixed_preds_test
                )
            
            axes[i+1].set_title(rf'$n_2 = {n_ec}$')
            plotting.plot_samples(axes[i+1], model.preds[-1], size)

    fig.savefig(f"outputs/compare_replay_{dataset}.png")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that compared the replays of PCNs with different number of EC neurons"
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
    cf.act_fn = utils.Tanh()
    cf.decay = False

    main(cf)