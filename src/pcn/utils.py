import random
import json
import numpy as np
import torch
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Activation(object):
    def forward(self, inp):
        raise NotImplementedError

    def deriv(self, inp):
        raise NotImplementedError

    def __call__(self, inp):
        return self.forward(inp)


class Linear(Activation):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return set_tensor(torch.ones((1,)))


class ReLU(Activation):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out


class Tanh(Activation):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0
    
class Sigmoid(Activation):
    def forward(self, inp):
        return torch.sigmoid(inp)
    
    def deriv(self, inp):
        return torch.sigmoid(inp)*(1 - torch.sigmoid(inp))


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_tensor(tensor):
    return tensor.to(DEVICE).float()


def flatten_array(array):
    return torch.flatten(torch.cat(array, dim=1))


def save_json(obj, path):
    with open(path, "w") as file:
        json.dump(obj, file)


def load_json(path):
    with open(path) as file:
        return json.load(file)
    
def calc_cov(model, dataloader, cf):
    errors = [[] for _ in range(model.n_nodes)]
    cov = []
    
    # Append neuron activities of all batches
    with torch.no_grad():
        for x, y in dataloader:
            x = set_tensor(x)
            model.test_batch(
                x, cf.n_test_iters, fixed_preds=cf.fixed_preds_train
            )            
            for n in range(model.n_nodes):
                errors[n] += model.errs[n].to('cpu').tolist()

    for n in range(model.n_nodes):
        e = torch.Tensor(errors[n])
        cov.append(torch.cov(e.T))

    return cov

# Reproducibility functions
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def recall_error(dataloader, model, n_iters=10000, step_tolerance=1e-5, fixed_preds=False): 
    errors = []
    for img_batch, _ in dataloader:
        img_batch_half = img_batch.clone()
        img_batch_half[:, 784//2:] = 0
        img_batch_half = set_tensor(img_batch_half)
        img_batch = set_tensor(img_batch)
        model.recall_batch(
                    img_batch_half, n_iters, step_tolerance, fixed_preds
        )
        errors.append(torch.sum((img_batch - model.mus[model.n_layers])**2, axis = 1)) 
    return np.mean(errors)


class EarlyStopping:
    def __init__(self, patience: int = 10, threshold: float = 1e-4):
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.early_stop = False
        self.num_bad_epochs = 0
        self.best_model_state = None

    def __call__(self, loss, model):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(loss)

        if self.best == None:
            self.best = current
            self.best_model_state = model.state_dict()

        elif self.is_better(current, self.best):
            self.best = current
            self.best_model_state = model.state_dict()
            self.num_bad_epochs = 0
        
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.early_stop = True
                self.num_bad_epochs = 0     

    def is_better(self, a, best):
        rel_epsilon = 1.0 - self.threshold
        return a < best * rel_epsilon

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)