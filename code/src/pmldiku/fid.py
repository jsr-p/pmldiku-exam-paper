from typing import Any, Callable, Dict, List

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.utils import save_image
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor

import pmldiku
from pmldiku import data, model_utils, output_utils, fid

# Frechet Inception Distance

class FID():
    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def embed(self, images) -> torch.Tensor:
        """takes final layer as"""
        embeddings = self.classifier.model.embed(images).detach()
        return embeddings

    def calculate_fid(self, real: torch.Tensor, generated: torch.Tensor) -> tuple[float, dict[str, torch.Tensor]]:
        """real: real images, generated: generated images"""
        embed_real = self.embed(real)
        embed_gen = self.embed(generated)

        mu_real, mu_gen = embed_real.mean(0), embed_gen.mean(0) 
        cov_real, cov_gen = embed_real.T.cov(), embed_gen.T.cov()

        score = self.fid_score(mu_real, mu_gen, cov_real, cov_gen)
        return score, {'mu_real': mu_real, 'mu_gen': mu_gen, 'cov_real': cov_real, 'cov_gen': cov_gen}

    @staticmethod
    def fid_score(mu_real, mu_gen, cov_real, cov_gen, n=None) -> float:     
        return torch.linalg.vector_norm(mu_real - mu_gen)**2 + (cov_real + cov_gen - 2*(cov_real @ cov_gen)**(1/2)).trace() 


# classifier

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def old_forward(self, x) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward(self, x) -> torch.Tensor:
        x = self.embed(x)
        return F.log_softmax(x)

    def embed(self, x) -> torch.Tensor:
        # generates embedding layer
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# callbacks
class ClassifierLossCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self) -> None:
        super().__init__()
        self.train_loss: list = []
        self.val_loss: list  = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.train_loss.append(trainer.callback_metrics['train_mse'].item())

class MNISTClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super(MNISTClassifier, self).__init__()

        # init a pretrained resnet
        self.model = Net()
        self.verbose = True
        
    def training_step(self, batch, _):
        loss = self.inner_step(batch, step_name="train")
        self.log(f"train_mse", loss)        
        return loss

    def validation_step(self, batch, _):
        loss = self.inner_step(batch, step_name="val")
        self.log(f"val_mse", loss)        
        return loss

    def inner_step(self, batch, step_name, **kwargs):
        X, y = batch
        y_hat = self.model(X)
        loss = F.nll_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer

    def training_epoch_end(self, training_step_outputs) -> None:
        if self.verbose:
            last_outputs = training_step_outputs[-50: -1]
            res = np.mean([d['loss'].cpu().numpy() for d in last_outputs])
            print(f"New Epoch. Loss: {res}")


#Utils

def avg_correct_label(y_hat: torch.Tensor, y:torch.Tensor) -> float:
    f: Callable = lambda x: 1 if x == True else 0
    bools: torch.Tensor = (y_hat == y)
    return np.mean([f(y_i) for y_i in bools])