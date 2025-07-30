import torch
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split

def transform_data(dataset):
    return {
       'images': dataset.data.float().view(-1, 784)/255.,
       'labels': dataset.targets.float()
       }

def make_mnist(data_path):
    """
    Save train and test datasets for MNIST
    """
    dset_train = MNIST(data_path, train=True,  download=False)
    dset_test  = MNIST(data_path, train=False, download=False)
    mnist_train = transform_data(dset_train)
    mnist_test = transform_data(dset_test)
    x_train = mnist_train['images']
    y_train = mnist_train['labels']
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=len(mnist_test['labels']), stratify=y_train, random_state=0)
    mnist_train = {'images': x_train, 'labels': y_train}
    mnist_valid = {'images': x_valid, 'labels': y_valid}
    torch.save(mnist_train, f'{data_path}/mnist_train.pt')
    torch.save(mnist_valid, f'{data_path}/mnist_valid.pt')
    torch.save(mnist_test, f'{data_path}/mnist_test.pt')
   
if __name__ == "__main__":
    data_path = "data"
    make_mnist(data_path)