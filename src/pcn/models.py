from pcn import utils
from pcn.layers import FCLayer
import torch
from torch import nn
import numpy as np
import wandb

class PCModel(object):
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

    def train_batch(self, img_batch, n_iters, layers_in_progress, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads(layers_in_progress)
    
    def test_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)

    def replay_batch(self, img_batch, label_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False, train=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.set_target(img_batch)
        self.replay_updates(n_iters, step_tolerance, fixed_preds, train) 

    def replay_updates(self, n_iters, step_tolerance, fixed_preds=False, train=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]        
        self.errs[self.n_nodes - 1] = utils.set_tensor(torch.zeros(self.mus[self.n_nodes - 1].shape))        
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        while not stop and itr <= n_iters: 
            for l in range(1, self.n_layers): # mus[-1] and mus[0] are fixed
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            self.errs[self.n_nodes - 1] = utils.set_tensor(torch.zeros(self.mus[self.n_nodes - 1].shape))   

            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n])      
            
            if train:
                self.update_grads()

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1         
        
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

    def test_convergence(self, img_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates_to_convergence(n_iters, step_tolerance, fixed_preds=fixed_preds)
        
    def updates_to_convergence(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers, batch_size)
        while not stop and itr <= n_iters:            
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta                

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n].item())

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1

    def recall_batch(self, img_batch_corrupt, n_iters, step_tolerance=1e-6, init_std=0.05, fixed_preds=False):
        batch_size = img_batch_corrupt.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch_corrupt)
        self.recall_updates(n_iters, step_tolerance, fixed_preds=fixed_preds)
        
    def recall_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers, batch_size)
        while not stop and itr <= n_iters:            
            for l in range(self.n_layers): 
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta       

            # Recall pixels
            delta = - self.errs[self.n_layers]
            self.mus[self.n_layers][:, 784//2:] = self.mus[self.n_layers][:, 784//2:] + self.mu_dt * delta[:, 784//2:]

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for m in range(batch_size):
                for n in range(self.n_nodes):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n].item())


            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1            

    def update_grads(self, layers_in_progress):
        for l in layers_in_progress:
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()
    
    def get_error_lengths(self):
        batch_size = len(self.errs[0])
        errors = torch.empty(batch_size, self.n_nodes)
        for n in range(self.n_nodes):
            errors[:, n]  = torch.sum(self.errs[n] ** 2)/self.nodes[n]
        return errors
    
    def get_weight_lengths(self):
        weights = torch.empty(self.n_layers)
        for l in range(self.n_layers):
            weights[l]  = torch.sum(self.layers[l].weights.flatten() ** 2)/(self.nodes[l]*self.nodes[l+1])
        return weights
    
    def get_latent_lengths(self):
        batch_size = len(self.errs[0])
        latents = torch.empty(batch_size, self.n_nodes - 1)
        for n in range(self.n_nodes - 1):
            latents[:, n]  = torch.sum(self.mus[n] ** 2)/self.nodes[n] 
        return latents
    
    def get_latents(self):
        return self.mus[:-1]

    @property
    def params(self):
        return self.layers
    
class iPCModel(object):
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

    def train_batch(self, img_batch, n_iters, layers_in_progress, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, layers_in_progress, fixed_preds=fixed_preds)
    
    def test_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, layers_in_progress = [], fixed_preds=fixed_preds)
    
    def replay_batch(self, img_batch, label_batch, n_iters, step_tolerance=1e-6, init_std=0.05, fixed_preds=False, train=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.set_target(img_batch)
        self.replay_updates(n_iters, step_tolerance, fixed_preds, train) 

    def replay_updates(self, n_iters, step_tolerance, fixed_preds=False, train=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]        
        self.errs[self.n_nodes - 1] = utils.set_tensor(torch.zeros(self.mus[self.n_nodes - 1].shape))        
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        while not stop and itr <= n_iters: 
            for l in range(1, self.n_layers): # mus[-1] and mus[0] are fixed
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            self.errs[self.n_nodes - 1] = utils.set_tensor(torch.zeros(self.mus[self.n_nodes - 1].shape))   

            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n])      
            
            if train:
                self.update_grads()

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1       
        
    def updates(self, n_iters, layers_in_progress, fixed_preds=False, train=True):
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
            
            self.update_grads(layers_in_progress)

    def test_convergence(self, img_batch, n_iters, step_tolerance=1e-6, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates_to_convergence(n_iters, step_tolerance, fixed_preds=fixed_preds)
        
    def updates_to_convergence(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers, batch_size)
        while not stop and itr <= n_iters:            
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta                

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n])

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1    

    def recall_batch(self, img_batch_corrupt, n_iters, step_tolerance=1e-6, init_std=0.05, fixed_preds=False, n_cut=784//2):
        batch_size = img_batch_corrupt.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch_corrupt)
        self.recall_updates(n_iters, step_tolerance, fixed_preds, n_cut)
        
    def recall_updates(self, n_iters, step_tolerance, fixed_preds=False, n_cut=784//2):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers, batch_size)
        while not stop and itr <= n_iters:
            for l in range(self.n_layers): 
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta       

            # Recall blank pixels
            delta = - self.errs[self.n_layers]
            self.mus[self.n_layers][:, n_cut:] = self.mus[self.n_layers][:, n_cut:] + self.mu_dt * delta[:, n_cut:]

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for m in range(batch_size):
                for n in range(self.n_nodes):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n].item())

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1       

    def update_grads(self, layers_in_progress):
        for l in layers_in_progress:
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()
    
    def get_error_lengths(self):
        batch_size = len(self.errs[0])
        errors = torch.empty(batch_size, self.n_nodes)
        for n in range(self.n_nodes):
            errors[:, n]  = torch.sum(self.errs[n] ** 2)/self.nodes[n] 
        return errors
    
    def get_weight_lengths(self):
        weights = torch.empty(self.n_layers)
        for l in range(self.n_layers):
            weights[l]  = torch.sum(self.layers[l].weights.flatten() ** 2)/(self.nodes[l]*self.nodes[l+1])
        return weights
    
    def get_latent_lengths(self):
        batch_size = len(self.errs[0])
        latents = torch.empty(batch_size, self.n_nodes - 1)
        for n in range(self.n_nodes - 1):
            latents[:, n]  = torch.sum(self.mus[n] ** 2)/self.nodes[n] 
        return latents
    
    def get_latents(self):
        return self.mus[:-1]

    @property
    def params(self):
        return self.layers
    
class PCModule(nn.Module):
    def __init__(self, nodes, act_fn, mu_dt, use_bias=False, kaiming_init=False):
        super().__init__()
        self.nodes = nodes
        self.mu_dt = mu_dt
        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = nn.Parameter(utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            ))

    def set_target(self, target):
        self.mus[-1] = target.clone()
    
    def set_input(self, inp):
        self.mus[0] = inp.clone()

    def get_errors(self):
        batch_size = len(self.errs[0])
        errors = torch.empty(batch_size, self.n_nodes)
        for n in range(self.n_nodes):
            errors[:, n]  = torch.sum(self.errs[n] ** 2)/self.nodes[n] 
        return errors
    
    def get_loss(self):
        # Average loss for a minibatch, normalized by the number of nodes per layer
        errors = self.get_errors()
        return errors.mean(axis=0).sum()

    def forward(self, img_batch, n_iters, label_batch=None, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        if label_batch is not None:
            self.set_input(label_batch)
        self.set_target(img_batch)
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
        return self.mus # forward function of a module has to return something   

    def forward_test(self, img_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.n_layers, batch_size)
        while not stop and itr <= n_iters:            
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta                

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for n in range(self.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(self.get_errors()[m, n])

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])
    
class PCTrainer(object):
    def __init__(self, model, optimizers):
        self.model = model
        self.optimizers = optimizers
    
    def train(self, train_loader, n_train_iters, fixed_preds_train):
        self.activations = [[] for n in range(self.model.n_nodes)]
        training_epoch_errors = [[] for _ in range(self.model.n_nodes)]
        batch_size = train_loader.batch_size
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):   
            img_batch = utils.set_tensor(img_batch)
            self.model(img_batch, n_train_iters, fixed_preds=fixed_preds_train)
            self.model.update_grads()

            for optim in self.optimizers:
                optim.step()
            errors = self.model.get_errors()
            
            # gather data for the current batch
            for n in range(self.model.n_nodes):
                training_epoch_errors[n] += [errors[:, n].mean().item()]
            
            # log layer activations (except input)
            step = batch_id*batch_size
            for n in range(self.model.n_nodes - 1):
                wandb.log({f'latents_{n}_train': wandb.Histogram(self.model.mus[n])}, step=step)

        # gather data for the full epoch
        training_errors = []
        for n in range(self.model.n_nodes):
            error = np.mean(training_epoch_errors[n])
            training_errors.append(error)
        return training_errors 
    
    def eval(self, valid_loader, n_test_iters, fixed_preds_test):
        img_batch, label_batch = next(iter(valid_loader))
        img_batch = utils.set_tensor(img_batch)
        self.model.forward_test(img_batch, n_test_iters, fixed_preds=fixed_preds_test)
        errors = self.model.get_errors()
        validation_errors = []
        for n in range(self.model.n_nodes):
            validation_errors.append(errors[:, n].mean().item())
        return img_batch, label_batch, validation_errors
    
    
class PCModule_auto(nn.Module):
    def __init__(self, nodes, act_fn, use_bias=False, kaiming_init=False):
        super().__init__()
        self.nodes = nodes

        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def get_target_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()
    
    def get_loss(self):
        return self.get_errors().sum(axis=0).mean()
    
    def get_errors(self):
        batch_size = len(self.errs[0])
        errors = torch.empty(batch_size, self.n_nodes)
        for n in range(self.n_nodes):
            errors[:, n]  = torch.sum(self.errs[n] ** 2)/self.nodes[n] 
        return errors
    
    def get_latents(self):
        return self.mus[:-1]
    
class PCTrainer_auto(object):
    def __init__(self, model, optimizer_mu_fn, optimizer_p_fn, mu_dt, lr):
        self.model = model
        self.mu_dt = mu_dt
        self.lr = lr
        self.optimizer_mu_fn = optimizer_mu_fn
        self.optimizer_p_fn = optimizer_p_fn
        self.optimizer_p = self.optimizer_p_fn(self.model.parameters(), self.lr)

    def reset(self):
        self.model.preds = [[] for _ in range(self.model.n_nodes)]
        self.model.errs = [[] for _ in range(self.model.n_nodes)]
        self.model.mus = [[] for _ in range(self.model.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.model.n_layers):
            self.model.mus[l] = nn.Parameter(utils.set_tensor(
                torch.empty(batch_size, self.model.layers[l].in_size).normal_(mean=0, std=init_std)
            ))

    def set_target(self, target):
        self.model.mus[-1] = target.clone()
    
    def set_input(self, inp):
        self.model.mus[0] = inp.clone()
    
    def train(self, train_loader, n_train_iters, fixed_preds_train):
        training_epoch_errors = [[] for _ in range(self.model.n_nodes)]
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):   
            img_batch = utils.set_tensor(img_batch)
            self.train_batch(
                img_batch, n_train_iters, fixed_preds=fixed_preds_train
            )
            self.optimizer_p.step()
            errors = self.model.get_errors()

            # gather data for the current batch
            for n in range(self.model.n_nodes):
                training_epoch_errors[n] += [errors[:, n].mean().item()]
            
        # gather data for the full epoch
        training_errors = []
        for n in range(self.model.n_nodes):
            error = np.mean(training_epoch_errors[n])
            training_errors.append(error)
        return training_errors 
    
    def eval(self, valid_loader, n_test_iters, fixed_preds_test):
        # Validation on a single batch (requires grad for inference)
        img_batch, label_batch = next(iter(valid_loader))
        img_batch = utils.set_tensor(img_batch)
        self.test_batch(
            img_batch, n_test_iters, fixed_preds=fixed_preds_test
        )
        errors = self.model.get_errors()
        validation_errors = []
        for n in range(self.model.n_nodes):
            validation_errors.append(errors[:, n].mean().item())
        return img_batch, label_batch, validation_errors
        
    def train_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()
    
    def test_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
    
    def replay(self, img_batch, label_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
        
    def updates(self, n_iters, fixed_preds=False):
        # Requires self.reset_mus() to get latents as parameters to be optimized
        self.optimizer_mu = self.optimizer_mu_fn(self.model.get_latents(), self.mu_dt)
        self.model.preds[0] = utils.set_tensor(torch.zeros(self.model.mus[0].shape))
        self.model.errs[0] = self.model.mus[0] - self.model.preds[0]
        for n in range(1, self.model.n_nodes):
            self.model.preds[n] = self.model.layers[n - 1].forward(self.model.mus[n - 1])
            self.model.errs[n] = self.model.mus[n] - self.model.preds[n]        

        for itr in range(n_iters):
            loss = self.model.get_loss()
            self.optimizer_mu.zero_grad()
            loss.backward()
            self.optimizer_mu.step()

            for n in range(1, self.model.n_nodes):
                if not fixed_preds:
                    self.model.preds[n] = self.model.layers[n - 1].forward(self.model.mus[n - 1])
                self.model.errs[n] = self.model.mus[n] - self.model.preds[n]

    def test_convergence(self, img_batch, n_iters, step_tolerance=1e-6, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates_to_convergence(n_iters, step_tolerance, fixed_preds=fixed_preds)
        
    def updates_to_convergence(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.model.mus[0].shape[0]
        self.plot_batch_errors = [[[] for m in range(batch_size)] for n in range(self.n_nodes)]
        self.model.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape))
        self.model.errs[0] = self.model.mus[0] - self.model.preds[0]
        for n in range(1, self.model.n_nodes):
            self.model.preds[n] = self.model.layers[n - 1].forward(self.mus[n - 1])
            self.model.errs[n] = self.model.mus[n] - self.model.preds[n]
        itr = 0
        stop = False
        relative_diff = torch.empty(self.model.n_layers, batch_size)
        while not stop and itr <= n_iters:            
            loss = self.model.get_loss()
            self.optimizer_mu.zero_grad()
            loss.backward()
            self.optimizer_mu.step()              

            for n in range(1, self.model.n_nodes):
                if not fixed_preds:
                    self.model.preds[n] = self.model.layers[n - 1].forward(self.model.mus[n - 1])
                self.model.errs[n] = self.model.mus[n] - self.model.preds[n]
            
            for n in range(self.model.n_nodes):
                for m in range(batch_size):
                    self.plot_batch_errors[n][m].append(self.model.get_errors()[n, m])

            stop = (relative_diff < step_tolerance).sum().item()
            itr += 1        

    def update_grads(self):
        loss = self.model.get_loss()
        self.optimizer_p.zero_grad()
        loss.backward()
        

