from typing import Protocol
import pytorch_lightning as pl
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image


class BaseVAE(pl.LightningModule):
    def __init__(self):
        super(BaseVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, 2)  # Latent space of 2D
        self.fc22 = nn.Linear(100, 2)  # Latent space of 2D
        self.fc3 = nn.Linear(2, 100)  # Latent space of 2D
        self.fc3a = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc1a(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def neg_elbo(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD  # -ELBO


class LitBaseVAE(pl.LightningModule):
    # def __init__(self, base_vae: BaseVAE):
    def __init__(self, base_vae: BaseVAE):
        super().__init__()
        # self.base_vae = base_vae
        self.base_vae = base_vae

    def training_step(self, batch, _):
        return self.inner_step(batch, step_name="train", prog_bar=True)

    def validation_step(self, batch, _):
        return self.inner_step(batch, step_name="val")

    def inner_step(self, batch, step_name: str, **kwargs):
        """Inner step of both train and validation steps"""
        x, _ = batch  # Don't use target labels
        recon_batch, mu, logvar = self.base_vae(x)
        loss = neg_elbo(recon_batch, x, mu, logvar)
        self.log(f"{step_name}_loss", loss, **kwargs)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class VAELossCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.val_loss = []
        self.train_loss = []

    def on_validation_end(self, trainer, _):
        self.val_loss.append(trainer.callback_metrics["val_loss"].item())

    def on_train_end(self, trainer, _):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())


class VAEImageReconstructionCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, num_epochs: int, save: bool = False):
        super().__init__()
        self.samples = torch.zeros(size=(num_epochs, 64, 28, 28))
        self.save = save
        self.epoch = 0

    def on_train_epoch_end(self, _, pl_module):
        with torch.no_grad():
            pl_module.eval()
            sample = torch.randn(64, 2).to(pl_module.device)
            sample = pl_module.base_vae.decode(sample)
            self.samples[self.epoch] = sample.view(64, 28, 28)
            pl_module.train()
            if self.save:
                save_image(sample.view(64, 1, 28, 28),
                           'results/sample_' + str(self.epoch) + '.png')
        self.epoch += 1


class Encoder(Protocol):
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        ...


def encode_means(
    encoder: Encoder, test_loader: torch.utils.data.DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Encodes test data into tensor of means with corresponding labels.

    Args:
        encoder: Encoder module
        test_loader: test dataloader
    Returns:
        tuple encoded means & labels
    """
    labels, all_encoded_means = [], []
    for X_b, y_b in test_loader:
        encoded_means, _ = encoder.encode(X_b.view(-1, 784))
        all_encoded_means.append(encoded_means)
        labels.append(y_b)
    means_plot = torch.concatenate(all_encoded_means).detach().numpy()
    labels = torch.concatenate(labels).numpy()
    return means_plot, labels
