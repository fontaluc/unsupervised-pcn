import numpy as np
import torch
import wandb

def get_optim(params, optim_id, lr, q_lr=None, batch_scale=True, grad_clip=None, weight_decay=None):
    if optim_id == "Adam":
        return Adam(
            params, lr=lr, q_lr=q_lr, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay
        )
    elif optim_id == "SGD":
        return SGD(
            params, lr=lr, q_lr=q_lr, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"{optim_id} not a valid optimizer ID")


class Optimizer:
    def __init__(self, params, batch_scale=True, grad_clip=None, weight_decay=None):
        self._params = params
        self.batch_scale = batch_scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

    def scale_batch(self, param, batch_size):
        if self.batch_scale:
            param.grad["weights"] = (1 / batch_size) * param.grad["weights"]
            if param.use_bias:
                param.grad["bias"] = (1 / batch_size) * param.grad["bias"]

    def clip_grads(self, param):
        if self.grad_clip is not None:
            param.grad["weights"] = torch.clamp(param.grad["weights"], -self.grad_clip, self.grad_clip)
            if param.use_bias:
                param.grad["bias"] = torch.clamp(param.grad["bias"], -self.grad_clip, self.grad_clip)

    def decay_weights(self, param):
        if self.weight_decay is not None:
            param.grad["weights"] = param.grad["weights"] - self.weight_decay * param.weights

    def step(self, *args, **kwargs):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr, q_lr=None, batch_scale=True, grad_clip=None, weight_decay=None):
        super().__init__(params, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay)
        self.lr = lr
        self.q_lr = q_lr

    def step(self, *args, batch_size=None, **kwargs):
        for param in self._params:
                # Update parameters if gradients of bias (if it is used) and weights are not None
                if param.grad["weights"] is not None and (not param.use_bias or param.grad["bias"] is not None):
                    _lr = self.q_lr if param.is_forward else self.lr
                    self.scale_batch(param, batch_size)
                    self.clip_grads(param)
                    self.decay_weights(param)

                    param.weights += _lr * param.grad["weights"]
                    if param.use_bias:
                        param.bias += _lr * param.grad["bias"]
                    param._reset_grad()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr,
        q_lr=None,
        batch_scale=True,
        eps=1e-8,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=None,
        grad_clip=None,
    ):
        super().__init__(params, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay)
        self.lr = lr
        self.q_lr = q_lr
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.c_b = [torch.zeros_like(param.bias) for param in self._params]
        self.c_w = [torch.zeros_like(param.weights) for param in self._params]
        self.v_b = [torch.zeros_like(param.bias) for param in self._params]
        self.v_w = [torch.zeros_like(param.weights) for param in self._params]

    def step(self, curr_epoch=None, curr_batch=None, n_batches=None, batch_size=None, log=False):
        with torch.no_grad():
            t = (curr_epoch) * n_batches + curr_batch

            for p, param in enumerate(self._params):
                if param.grad["weights"] is not None and (not param.use_bias or param.grad["bias"] is not None):
                    _lr = self.q_lr if param.is_forward else self.lr
                    self.scale_batch(param, batch_size)
                    self.clip_grads(param)
                    self.decay_weights(param)

                    # Log clipped gradients
                    if log:
                        wandb.log({f'grad_{p}': wandb.Histogram(param.grad['weights'].cpu().detach())})
                        if param.use_bias:
                            wandb.log({f'grad_b_{p}': wandb.Histogram(param.grad['bias'].cpu().detach())})

                    self.c_w[p] = self.beta_1 * self.c_w[p] + (1 - self.beta_1) * param.grad["weights"]
                    self.v_w[p] = self.beta_2 * self.v_w[p] + (1 - self.beta_2) * param.grad["weights"] ** 2
                    delta_w = np.sqrt(1 - self.beta_2 ** t) * self.c_w[p] / (torch.sqrt(self.v_w[p]) + self.eps)
                    param.weights += _lr * delta_w

                    if param.use_bias:
                        self.c_b[p] = self.beta_1 * self.c_b[p] + (1 - self.beta_1) * param.grad["bias"]
                        self.v_b[p] = self.beta_2 * self.v_b[p] + (1 - self.beta_2) * param.grad["bias"] ** 2
                        delta_b = (
                            np.sqrt(1 - self.beta_2 ** t) * self.c_b[p] / (torch.sqrt(self.v_b[p]) + self.eps)
                        )
                        param.bias += _lr * delta_b

                    param._reset_grad()

class LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving if it is sufficiently low.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): The number of allowed epochs with no improvement after
            which the learning rate will be reduced.
            For example, consider the case of having no patience (`patience = 0`).
            In the first epoch, a baseline is established and is always considered good as there's no previous baseline.
            In the second epoch, if the performance is worse than the baseline,
            we have what is considered an intolerable epoch.
            Since the count of intolerable epochs (1) is greater than the patience level (0),
            the learning rate is reduced at the end of this epoch.
            From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
            if the performance is worse than the baseline. If the performance improves or remains the same,
            the learning rate is not adjusted.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        min_lr (float): A lower bound on the learning rate. Default: 0.

    Adapted from Pytorch.
    """

    def __init__(
            self, 
            optimizer: Optimizer,
            factor: float = 0.1,
            patience: int = 10,
            threshold: float = 1e-4,
            low_threshold: float = 0.2,
            min_lr: float = 0
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.previous = None
        self.num_bad_epochs = 0   
        self.max = None
        self.low_threshold = low_threshold
        self.min_lr = min_lr
        
    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.max == None:
            self.max = current

        elif self.is_better(current, self.previous):
            self.num_bad_epochs = 0
            
        elif current > self.max:
            self.max = current

        elif self.is_low(current, self.max):
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0

        else:
            self.num_bad_epochs = 0
        
        self.previous = current

    def _reduce_lr(self):
        self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)

    def is_better(self, a, best):
        rel_epsilon = 1.0 - self.threshold
        return a < best * rel_epsilon
    
    def is_low(self, a, max):
        return a < self.low_threshold*max
    
class ExponentialLR(LRScheduler):
    """Decays the learning rate by gamma every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float
    ):  
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr = self.optimizer.lr * self.gamma