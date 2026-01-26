from torch.utils import data
from torchvision import datasets, transforms

from pcn import utils


class MNIST(datasets.MNIST):
    def __init__(self, train, size=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.1307), std=(0.3081))
        super().__init__("./data/mnist", download=True, transform=transform, train=train)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, size=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        super().__init__("./data/cifar10", download=True, transform=transform, train=train)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index): # get a single sample
        data, target = super().__getitem__(index)
        data = _to_vector(data) 
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, size=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5), std=(0.5))
        super().__init__("./data/fmnist", download=True, transform=transform, train=train)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


def get_dataloader(dataset, batch_size, worker_init_fn, generator): 
    # make batches of samples individually obtained with __getitem__
    dataloader = data.DataLoader(
        dataset, 
        batch_size, 
        shuffle=True, 
        drop_last=True, 
        worker_init_fn=worker_init_fn, 
        generator=generator) 
    return list(map(_preprocess_batch, dataloader)) # list of (data, label) batches

def _preprocess_batch(batch):
    batch[0] = utils.set_tensor(batch[0])
    return (batch[0], batch[1])


def _get_transform(normalize=True, mean=(0.5), std=(0.5)):
    transform = [transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform)

def _to_vector(data):
    return data.flatten()