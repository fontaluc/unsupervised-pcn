import argparse
from tqdm import tqdm
from pcn import utils, plotting
import torch
from torch.utils.data import DataLoader, TensorDataset
from pcn.models import PCModel, PCTrainer
import os
from sklearn.linear_model import LogisticRegression

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

    dataset_valid = torch.load('data/mnist_valid.pt')
    dset_valid = TensorDataset(dataset_valid['images'], dataset_valid['labels'])
    valid_loader  = DataLoader(
        dset_valid, 
        batch_size=cf.batch_size, 
        shuffle=True, 
        worker_init_fn=utils.seed_worker, 
        generator=g
    )
        
    model_name = f"pcn-n_vc={cf.n_vc}-n_ec={cf.n_ec}"
    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init, use_decay=cf.decay
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    # Create folder if it doesn't exist
    if not os.path.exists(f"outputs/{model_name}"):
        os.makedirs(f"outputs/{model_name}")

    ## Recall performance
    # img_batch, label_batch = next(iter(train_loader))
    # n_cut = img_batch.size(1)//2
    # # Quantitatively on the whole training set: average RMSE between recalled and original images
    # recall_rmse = 0 
    # with torch.no_grad():
    #     for img_batch, label_batch in tqdm(train_loader):
    #         img_batch_half = utils.mask_image(img_batch, n_cut)
    #         img_batch_half = utils.set_tensor(img_batch_half)
    #         model.recall_batch(
    #             img_batch_half, 
    #             cf.n_max_iters, 
    #             n_cut=n_cut, 
    #             step_tolerance=cf.step_tolerance,
    #             init_std=cf.init_std,
    #             fixed_preds=cf.fixed_preds_test
    #         )
    #         img_batch = utils.set_tensor(img_batch)
    #         recall_rmse += torch.sum(utils.rmse(img_batch, model.mus[-1])).item()
    # recall_rmse = recall_rmse/N

    ## Generalization performance
    trainer = PCTrainer(model)
    with torch.no_grad():
        valid_rmse = trainer.test(valid_loader, cf.n_max_iters, cf.fixed_preds_test)

    # Episodic replay RMSE  
    replay_rmse = 0
    with torch.no_grad():
        for img_batch, label_batch in tqdm(train_loader):
            img_batch = utils.set_tensor(img_batch)
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
            replay_rmse += torch.sum(utils.rmse(img_batch, model.preds[-1])).item()
    replay_rmse = replay_rmse/N

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

    mode = 'a' if os.path.exists("outputs/tune_second_layer.txt") else 'w'
    with open("outputs/tune_second_layer.txt", mode) as f:
        f.write(f"{cf.n_ec}, {replay_rmse}, {valid_rmse}, {valid_acc} \n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluate the PC model according to different metrics to choose the right number of EC units"
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
    cf.n_ec = args.n_ec
    cf.nodes = [cf.n_ec, 400, 784]
    cf.use_bias = True
    cf.kaiming_init = False
    cf.decay = False
    cf.act_fn = utils.Tanh()

    main(cf)