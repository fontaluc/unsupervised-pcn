import torch
from torch import nn
from pcn import utils
import math

class Layer(nn.Module):
    def __init__(
        self, in_size, out_size, act_fn, use_bias=False, kaiming_init=False, is_forward=False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.use_bias = use_bias
        self.is_forward = is_forward
        self.kaiming_init = kaiming_init

        self.weights = None
        self.bias = None
        self.grad = {"weights": None, "bias": None}

        if kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        if self.kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_params(self):
        weights = torch.empty((self.in_size, self.out_size)).normal_(mean=0.0, std=0.05)
        bias = torch.zeros((self.out_size))
        self.weights = nn.Parameter(utils.set_tensor(weights))
        self.bias = nn.Parameter(utils.set_tensor(bias))

    def _reset_params_kaiming(self):
        self.weights = nn.Parameter(utils.set_tensor(torch.empty((self.in_size, self.out_size))))
        self.bias = nn.Parameter(utils.set_tensor(torch.zeros((self.out_size))))
        if isinstance(self.act_fn, utils.Linear):
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        elif isinstance(self.act_fn, utils.Tanh):
            nn.init.kaiming_normal_(self.weights)
        elif isinstance(self.act_fn, utils.ReLU):
            nn.init.kaiming_normal_(self.weights)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

class FCLayer(Layer):
    def __init__(
        self, 
        in_size, 
        out_size, 
        act_fn, 
        use_bias=False, 
        kaiming_init=False, 
        is_forward=False, 
        use_decay=False, 
        alpha=0.1, 
        ema_alpha=0.01
    ):
        super().__init__(
            in_size, 
            out_size, 
            act_fn, 
            use_bias, 
            kaiming_init,
            is_forward=is_forward)
        self.use_bias = use_bias
        self.inp = None
        self.use_decay = use_decay
        self.alpha = alpha
        self.ema_alpha = ema_alpha
        self.theta_meta = 0

    def forward(self, inp):
        self.inp = inp.clone()
        out = self.act_fn(torch.matmul(self.inp, self.weights))
        if self.use_bias:
            out = out + self.bias
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights))
        out = torch.matmul(err * fn_deriv, self.weights.T)
        return out

    def update_gradient(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights))
        delta = torch.matmul(self.inp.T, err * fn_deriv)        
        if self.use_decay:
            activity = torch.mean(self.inp ** 2)
            self.theta_meta = (1 - self.ema_alpha) * self.theta_meta + self.ema_alpha * activity
            if activity > self.theta_meta:
                delta *= self.alpha
        self.grad["weights"] = delta
        if self.use_bias:
            self.grad["bias"] = torch.sum(err, axis=0)
    