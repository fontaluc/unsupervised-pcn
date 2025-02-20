from pcn import utils
from pcn.layers import FCLayer
import torch

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
            errors[:, n]  = torch.sum(self.errs[n] ** 2, axis=1)/self.nodes[n] 
        return errors
    
    def get_weight_lengths(self):
        weights = torch.empty(self.n_layers)
        for l in range(self.n_layers):
            weights[l]  = self.layers[l].weights.sum()/(self.nodes[l]*self.nodes[l+1])
        return weights
    
    def get_latent_lengths(self):
        batch_size = len(self.errs[0])
        latents = torch.empty(batch_size, self.n_nodes - 1)
        for n in range(self.n_nodes - 1):
            latents[:, n]  = torch.sum(self.mus[n] ** 2, axis=1)/self.nodes[n] 
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
            errors[:, n]  = torch.sum(self.errs[n] ** 2, axis=1)/self.nodes[n] 
        return errors
    
    def get_weight_lengths(self):
        weights = torch.empty(self.n_layers)
        for l in range(self.n_layers):
            weights[l]  = self.layers[l].weights.sum()/(self.nodes[l]*self.nodes[l+1])
        return weights
    
    def get_latent_lengths(self):
        batch_size = len(self.errs[0])
        latents = torch.empty(batch_size, self.n_nodes - 1)
        for n in range(self.n_nodes - 1):
            latents[:, n]  = torch.sum(self.mus[n] ** 2, axis=1)/self.nodes[n] 
        return latents
    
    def get_latents(self):
        return self.mus[:-1]

    @property
    def params(self):
        return self.layers