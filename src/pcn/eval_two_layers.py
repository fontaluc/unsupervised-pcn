import argparse
from tqdm import tqdm
from pcn import utils, plotting, datasets
import torch
from pcn.models import PCModel, PCTrainer
import os
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.normalize)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size)

    if cf.dataset == 'mnist':
        cf.n_vc = 450
    elif cf.dataset == 'fmnist':
        cf.n_vc = 750
    else:
        cf.n_vc = 2000

    nodes = [cf.n_ec, cf.n_vc, np.prod(size)]
        
    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}-n_ec={cf.n_ec}" if cf.n_ec > 0 else f"pcn-{cf.dataset}-n_vc={cf.n_vc}"
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    # Create folder if it doesn't exist
    if not os.path.exists(f"outputs/{model_name}"):
        os.makedirs(f"outputs/{model_name}")

    # Episodic replay RMSE
    if cf.n_ec > 0:  
        replay_mse = 0
        with torch.no_grad():
            for img_batch, label_batch in tqdm(train_loader):
                # Get EC activities for episodes
                model.test_batch(
                    img_batch, 
                    n_iters=cf.n_max_iters, 
                    step_tolerance=cf.step_tolerance, 
                    init_std=cf.init_std, 
                    fixed_preds=cf.fixed_preds_test
                )
                ec_batch = utils.set_tensor(model.mus[0])
                model.replay_batch(
                    ec_batch, 
                    cf.n_max_iters,
                    step_tolerance=cf.step_tolerance,
                    init_std=cf.init_std,
                    fixed_preds=cf.fixed_preds_test
                )
                replay_mse += torch.sum(utils.mse(img_batch, model.preds[-1])).item()
        replay_mse = replay_mse/len(train_dataset)
    else:
        replay_mse = float("nan")

    ## Pattern completion performance
    indices = utils.get_indices(size)
    recall_mse = 0 
    with torch.no_grad():
        for img_batch, label_batch in tqdm(valid_loader):
            img_batch_half = utils.mask_image(img_batch, indices)
            img_batch_half = utils.set_tensor(img_batch_half)
            model.recall_batch(
                img_batch_half, 
                cf.n_max_iters, 
                indices, 
                step_tolerance=cf.step_tolerance,
                init_std=cf.init_std,
                fixed_preds=cf.fixed_preds_test
            )
            recall_mse += torch.sum(utils.mse(img_batch, model.mus[-1])).item()
    recall_mse = recall_mse/len(valid_dataset)

    ## Generalization performance
    trainer = PCTrainer(model)
    with torch.no_grad():
        valid_mse = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)

    # Classification accuracy
    activities_train, labels_train = plotting.infer_latents(
        model, train_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_train = activities_train[0]
    y_train = labels_train
    clf = LogisticRegression(random_state=cf.seed).fit(X_train, y_train)


    activities_valid, labels_valid = plotting.infer_latents(
        model, valid_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_valid = activities_valid[0]
    y_valid = labels_valid
    valid_acc = clf.score(X_valid, y_valid)

    data = [cf.dataset, cf.n_ec, replay_mse, recall_mse, valid_mse, valid_acc]
    if os.path.exists("outputs/eval_two_layers.csv"):
        df = pd.read_csv("outputs/eval_two_layers.csv")
        df.loc[len(df)] = data
    else:
        df = pd.DataFrame([data], columns=['Dataset', 'EC size', 'Replay error', 'Completion error', 'Validation error', 'Validation accuracy'])
    df.to_csv('outputs/eval_two_layers.csv', index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluate the PC model according to different metrics to choose the right number of EC units"
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_ec", type=int, default=30, help="Enter size of EC layer")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
    cf.dataset = args.dataset
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
    cf.n_ec = args.n_ec
    cf.use_bias = True
    cf.kaiming_init = False
    cf.act_fn = utils.Tanh()

    main(cf)
