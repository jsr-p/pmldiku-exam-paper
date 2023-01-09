from pathlib import Path

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

FP_DATA = Path(__file__).parents[2] / "data"
Path.mkdir(FP_DATA, exist_ok=True)


class MNISTWrapper:
    def __init__(self, mnist: MNIST) -> None:
        self.mnist = mnist

    def extract_data(self):
        X = self.mnist.data
        X = X.reshape(X.shape[0], -1).double()  # Reshape into 2d matrix
        y = self.mnist.targets
        return X, y

    def setup_data_loader(
        self, batch_size=128, use_cuda=False
    ) -> torch.utils.data.DataLoader:
        kwargs = {"num_workers": 1, "pin_memory": use_cuda}
        loader = torch.utils.data.DataLoader(
            dataset=self.mnist, batch_size=batch_size, shuffle=True, **kwargs
        )
        return loader


def load_mnist(train: bool = True) -> MNISTWrapper:
    trans = transforms.ToTensor()
    mnist = MNIST(root=str(FP_DATA), train=train, download=True, transform=trans)
    return MNISTWrapper(mnist)


if __name__ == "__main__":
    mnist_train = load_mnist(train=True)
    mnist_test = load_mnist(train=False)
    X_train, y_train = mnist_train.extract_data()
    X_test, y_test = mnist_test.extract_data()
    train_loader = mnist_train.setup_data_loader()
