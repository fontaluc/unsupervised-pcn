from pcn import utils
from pcn.layers import FCLayer
import torch
from torch import nn
import numpy as np
import wandb

class PCModel(nn.Module):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False):
        self.nodes = nodes
        self.mu_dt = mu_dt

        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        self.layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )

    def set_target(self, target):
        self.mus[-1] = target.clone()
    
    def set_input(self, inp):
        self.mus[0] = inp.clone()

    def train_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()
    
    def eval_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)

    def test_batch(self, img_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.test_updates(n_iters, step_tolerance, fixed_preds=fixed_preds)

    def replay_batch(self, label_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = label_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.replay_updates(n_iters, step_tolerance, fixed_preds)

    def recall_batch(self, img_batch_corrupt, n_iters, n_cut, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch_corrupt.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch_corrupt)
        self.recall_updates(n_iters, step_tolerance, n_cut, fixed_preds=fixed_preds)

    def updates(self, n_iters, fixed_preds=False):
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def replay_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes - 1):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]        
        self.errs[-1] = utils.set_tensor(torch.zeros(batch_size, self.layers[-1].out_size))        
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        for itr in range(n_iters): 
            for l in range(1, self.n_layers): # mus[-1] and mus[0] are fixed
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes - 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            errors = self.get_errors()
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m, n])          

            if (relative_diff < step_tolerance).sum().item():
                # Replay
                n = self.n_nodes - 1
                if not fixed_preds: 
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1]) 
                break
        
    def test_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        relative_diff = torch.empty(self.n_layers, batch_size)
        for itr in range(n_iters):           
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta                

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            errors = self.get_errors()
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m, n])

            if (relative_diff < step_tolerance).sum().item():
                break
        
    def recall_updates(self, n_iters, step_tolerance, n_cut, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        relative_diff = torch.empty(self.n_layers, batch_size)
        for itr in range(n_iters):           
            for l in range(self.n_layers): 
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta       
            # Recall pixels
            delta = - self.errs[-1]
            self.mus[-1][:, n_cut:] = self.mus[-1][:, n_cut:] + self.mu_dt * delta[:, n_cut:]

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            errors = self.get_errors()
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m, n])

            if (relative_diff < step_tolerance).sum().item():
                break       

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])
    
    def get_errors(self):
        batch_size = len(self.errs[0])
        errors = torch.empty(batch_size, self.n_nodes)
        for n in range(self.n_nodes):
            errors[:, n]  = torch.sum(self.errs[n] ** 2)/self.nodes[n]
        return errors
    
class PCTrainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def train(self, train_loader, epoch, n_train_iters, fixed_preds_train, log_freq):
        self.activations = [[] for n in range(self.model.n_nodes)]
        training_epoch_errors = [[] for _ in range(self.model.n_nodes)]
        n_batches = len(train_loader)
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):   
            img_batch = utils.set_tensor(img_batch)
            self.model.train_batch(img_batch, n_train_iters, fixed_preds=fixed_preds_train)

            # log gradients
            t = epoch * n_batches + batch_id
            if t%log_freq == 0:
                for l in range(self.model.n_layers):
                    wandb.log({f'grad_{l}': wandb.Histogram(self.model.layers[l].grad['weights'].cpu().detach())})

            self.optimizer.step(
                curr_epoch=epoch,
                curr_batch=batch_id,
                n_batches=n_batches,
                batch_size=img_batch.size(0),
            )
            errors = self.model.get_errors()
            
            # gather data for the current batch
            for n in range(self.model.n_nodes):
                training_epoch_errors[n] += [errors[:, n].mean().item()]
            
            # log layer activations (except input) and weights            
            if t%log_freq == 0:
                for n in range(self.model.n_nodes - 1):
                    wandb.log({f'latents_{n}': wandb.Histogram(self.model.mus[n].cpu().detach())})
                    wandb.log({f'weights_{n}': wandb.Histogram(self.model.layers[n].weights.cpu().detach())})

        # gather data for the full epoch
        training_errors = []
        for n in range(self.model.n_nodes):
            error = np.mean(training_epoch_errors[n])
            training_errors.append(error)
        return training_errors 
    
    def eval(self, valid_loader, n_test_iters, fixed_preds_test):
        img_batch, label_batch = next(iter(valid_loader))
        img_batch = utils.set_tensor(img_batch)
        self.model.eval_batch(img_batch, n_test_iters, fixed_preds=fixed_preds_test)
        errors = self.model.get_errors()
        validation_errors = []
        for n in range(self.model.n_nodes):
            validation_errors.append(errors[:, n].mean().item())
        return img_batch, label_batch, validation_errors