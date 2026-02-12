import torch
from torch import nn
from pcn import utils
import math

class Layer(nn.Module):
    def __init__(
        self, in_size, out_size, act_fn, use_bias=False, kaiming_init=False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.use_bias = use_bias
        self.kaiming_init = kaiming_init

        self.weights = nn.Parameter(utils.set_tensor(torch.empty((self.in_size, self.out_size))))
        self.bias = nn.Parameter(utils.set_tensor(torch.zeros((self.out_size))))
        self._reset_grad()

        if kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def reset(self):
        if self.kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_params(self):
        nn.init.normal_(self.weights, mean=0.0, std=0.05)

    def _reset_params_kaiming(self):        
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))         
        if self.use_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class FCPlusLayer(Layer):
    def __init__(
        self, 
        in_size, 
        out_size, 
        act_fn, 
        use_bias=False, 
        kaiming_init=False 
    ):
        super().__init__(
            in_size, 
            out_size, 
            act_fn, 
            use_bias, 
            kaiming_init)
        self.use_bias = use_bias
        self.theta_meta = 0
        self.inp = None

    def forward(self, inp):
        self.inp = inp.clone()
        out = torch.matmul(self.act_fn(self.inp), self.weights) + self.bias
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(self.inp)
        out = fn_deriv * torch.matmul(err, self.weights.T)
        return out

    def update_gradient(self, err):
        delta = torch.matmul(self.act_fn(self.inp).T, err)        
        self.grad["weights"] = delta
        if self.use_bias:
            self.grad["bias"] = torch.sum(err, axis=0)

class FCLayer(Layer):
    def __init__(
        self, 
        in_size, 
        out_size, 
        act_fn, 
        use_bias=False, 
        kaiming_init=False 
    ):
        super().__init__(
            in_size, 
            out_size, 
            act_fn, 
            use_bias, 
            kaiming_init)
        self.use_bias = use_bias
        self.theta_meta = 0
        self.inp = None

    def forward(self, inp):
        self.inp = inp.clone()
        out = self.act_fn(torch.matmul(self.inp, self.weights) + self.bias)
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights) + self.bias)
        out = torch.matmul(err * fn_deriv, self.weights.T)
        return out

    def update_gradient(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights) + self.bias)
        delta = torch.matmul(self.inp.T, err * fn_deriv)        
        self.grad["weights"] = delta
        if self.use_bias:
            self.grad["bias"] = torch.sum(err * fn_deriv, axis=0)
    