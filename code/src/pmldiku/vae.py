"""Module to implement different VAE architectures."""

from typing import Protocol, TypeAlias
import pytorch_lightning as pl
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.distributions as tdist

from pmldiku import model_utils

EncoderOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
VAEOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BaseVAE(pl.LightningModule):
    def __init__(self, hidden_dim: int = 2):
        super(BaseVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self._init_network()

    def _init_network(self):
        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, self.hidden_dim)
        self.fc22 = nn.Linear(100, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 100)
        self.fc3a = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """Encodes the input image tensor.

        The image is encoded into the mean and variance
        of the encoder distribution q(z | x).

        Args:
            x: (B, 784) dimensional image tensor

        Returns:
            (EncoderOutput):
                Mean and variance of encoder distribution.
        """
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc1a(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draws z ~ q(z | x) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes sample from the latent distribution.

        The sample from the latent distribution is decoded
        into an image tensor.

        Returns:
            (B, 784) tensor of decoded images.
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Forward pass of the VAE."""
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CVAE(pl.LightningModule):
    """Implementation of the convolutional variational autoencoder.

    Inspiration taken from:
        - https://www.tensorflow.org/tutorials/generative/cvae
    """

    def __init__(self, hidden_dim: int):
        super(CVAE, self).__init__()
        self.hidden_dim = hidden_dim

        # Init network
        self._init_encoder()
        self._encode_cv_dim()
        self.fc_mean = nn.Linear(self.total_dim_encoder, hidden_dim)
        self.fc_logvar = nn.Linear(self.total_dim_encoder, hidden_dim)
        self._init_decoder()

    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

    def _encode_cv_dim(self):
        """Computes the dimension of input img. after encoder layer."""
        dim_cv1 = model_utils.compute_outputdim_cv(I=28, F=3, P=1, S=2)
        self.img_dim_encoder = model_utils.compute_outputdim_cv(
            I=dim_cv1, F=3, P=1, S=2
        )
        self.total_dim_encoder = 64 * self.img_dim_encoder**2

    def _init_decoder(self):
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.total_dim_encoder),
            nn.Unflatten(
                dim=1, unflattened_size=[64, self.img_dim_encoder, self.img_dim_encoder]
            ),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                stride=(1, 1),
                # K = 4 to get output of dim (B, 28, 28), might not
                # be the best to do? Raschka trims from 29 -> 28,
                # but that feels even more hacky.
                kernel_size=(4, 4),
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor):
        mu, logvar = self.encode(X)
        Z = self.reparameterize(mu, logvar)
        return self.decode(Z).view(-1, 784), mu, logvar

    def encode(self, X: torch.Tensor) -> EncoderOutput:
        X = self.encoder(X)
        mu, logvar = self.fc_mean(X), self.fc_logvar(X)
        return mu, logvar

    def decode(self, Z: torch.Tensor):
        return self.decoder(Z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAELoss:
    """Class to construct loss function for VAE.

    Note:
        See slide 19
        - https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf
        - https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss
    """

    def __init__(self, logpx_loss: str):
        self.logpx_loss = logpx_loss
        self.set_loss_fn()

    def set_loss_fn(self):
        match self.logpx_loss:
            case "mse":
                self.logpx_loss_fn = self.mse_loss
            case "bce":
                self.logpx_loss_fn = self.bce_loss
            case "cb":
                self.logpx_loss_fn = self.cb_loss
            case _:
                raise ValueError("Only MSE, BCE and CB losses defined for log p(x | z)")

    def mse_loss(self, recon_x: torch.Tensor, x: torch.Tensor, reduction: str = "sum"):
        return F.mse_loss(recon_x, x, reduction=reduction)

    def cb_loss(self, recon_x: torch.Tensor, x: torch.Tensor):
        """Implements the loss function for continuous Bernoulli r.v.s.

        The loss function for the VAE with the assumption of 
        X | Z ~ CB (continuous Bernoulli) is equal to the loss 
        function of standard Bernoulli with an added constant term.
        See p. 4 here:
            https://arxiv.org/pdf/1907.06845.pdf
        """
        dist = tdist.ContinuousBernoulli(probs=recon_x)
        return self.bce_loss(recon_x, x) - dist._cont_bern_log_norm().sum()

    def bce_loss(self, recon_x: torch.Tensor, x: torch.Tensor, reduction: str = "sum"):
        return F.binary_cross_entropy(recon_x, x, reduction=reduction)


    def __call__(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """Loss function for VAE; the negative ELBO.

        Depending on the assumption of p(x | z) the second
        term of the loss function is either a BCE or MSE.

        Args:
            recon_x: tensor of dim (B, 784) of reconstructed image pixels.
            x: tensor of dim (B, 1, 28, 28) of original image pixels.
            mu: parametrized mean tensor
            logvar: parametrized logvar tensor

        Returns:
            (float): loss
        """
        logpx_loss = self.logpx_loss_fn(recon_x, x.view(-1, 784))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return logpx_loss + KLD


class VAE(Protocol):
    """Protocol for a VAE"""

    def __call__(self, X: torch.Tensor) -> VAEOutput:
        ...


class Encoder(Protocol):
    """Protocol for an encoder"""

    def encode(self, X: torch.Tensor) -> EncoderOutput:
        ...


class LitVAE(pl.LightningModule):
    def __init__(self, vae: VAE, logpx_loss: str = "bce"):
        super().__init__()
        self.vae = vae
        self.loss_fn = VAELoss(logpx_loss=logpx_loss)

    def training_step(self, batch, _):
        return self.inner_step(batch, step_name="train", prog_bar=True)

    def validation_step(self, batch, _):
        return self.inner_step(batch, step_name="val")

    def inner_step(self, batch, step_name: str, **kwargs):
        """Inner step of both train and validation steps"""
        x, _ = batch  # Don't use target labels
        recon_batch, mu, logvar = self.vae(x)
        loss = self.loss_fn(recon_batch, x, mu, logvar)
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

    def __init__(self, save: bool = False):
        super().__init__()
        self._samples = []
        self.save = save
        self.epoch = 0
        self.samples: torch.Tensor

    def on_train_epoch_end(self, _, pl_module):
        with torch.no_grad():
            pl_module.eval()
            sample = torch.randn(64, 2).to(pl_module.device)
            sample = pl_module.vae.decode(sample)
            self._samples.append(sample.view(64, 28, 28))
            pl_module.train()
            if self.save:
                save_image(
                    sample.view(64, 1, 28, 28),
                    "results/sample_" + str(self.epoch) + ".png",
                )
        self.epoch += 1

    def on_train_end(self, *_):
        self.samples = torch.stack(self._samples)


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
        encoded_means, _ = encoder.encode(X_b)
        all_encoded_means.append(encoded_means)
        labels.append(y_b)
    means_plot = torch.concatenate(all_encoded_means).detach().numpy()
    labels = torch.concatenate(labels).numpy()
    return means_plot, labels
