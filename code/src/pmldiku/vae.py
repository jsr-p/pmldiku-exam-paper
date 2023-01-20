"""Module that implements different VAE architectures."""

from collections import OrderedDict
from typing import Protocol, TypeAlias, Optional

import numpy as np
import tqdm
import pytorch_lightning as pl
import torch
import torch.distributions as tdist
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

import pmldiku
from pmldiku import model_utils

EncoderOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
VAEOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# --------------------- Basic VAE --------------------- #


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

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Forward pass of the VAE.

        Args:
            x: Image tensor of dim. (B, W, H)

        Returns:
            (VAEOutput):
                decoded image, encoder mean, encoder log(variance).
        """
        z, mu, logvar = self.sample(x.view(-1, 784))
        return self.decode(z), mu, logvar

    def sample(self, x: torch.Tensor):
        """Samples from the encoder distribution.

        Returns:
            sample, encoder mean, encoder log(variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """Encodes the input image tensor.

        The image is encoded into the mean and variance
        of the encoder distribution q(z | x).

        Args:
            x: (B, 784) dimensional image tensor

        Returns:
            (EncoderOutput):
                Mean and variance of encoder distribution.
                Both are of dimension (B, self.hidden_dim)
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


class BayesVAELinearParams:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.empty(self.in_features, self.out_features)
        self.bias = torch.empty(self.out_features)

    def __call__(self, input: torch.Tensor):
        return F.linear(input, weight=self.weights, bias=self.bias)

    def unpack(self, start_idx: int, theta: torch.Tensor) -> int:
        num_weight_params = self.num_weight_params()
        num_bias_params = self.num_bias_params()
        weights_idx = start_idx + num_weight_params
        bias_idx = weights_idx + num_bias_params
        weights = theta[start_idx:weights_idx]
        bias = theta[weights_idx:bias_idx]
        self.update_params(weights, bias)
        return bias_idx

    def update_params(self, weights: torch.Tensor, bias: torch.Tensor):
        """Updates the parameters with the slice from the drawn theta.

        Note:
            F.linear computes the linear transformation as
                y = x A^T + b
            thus we set the parameters as below.
        """
        self.weights = weights.view((self.out_features, self.in_features))
        self.bias = bias

    def total_params(self):
        return self.num_weight_params() + self.num_bias_params()

    def num_weight_params(self):
        return self.in_features * self.out_features

    def num_bias_params(self):
        return self.out_features


class BayesVAE(pl.LightningModule):
    def __init__(self, hidden_dim: int = 2):
        super(BayesVAE, self).__init__()
        self.hidden_dim = hidden_dim
        self._init_encoder()
        self._init_decoder()
        self._init_posterior_theta()
        # pytorch_lightning won't capture the params if we
        # intialize them inside `_init_posterior_theta`
        self.mu_theta = nn.Parameter(torch.randn(self.total_params_theta))
        self.logvar_theta = nn.Parameter(torch.randn(self.total_params_theta))

    def _init_encoder(self):
        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, self.hidden_dim)
        self.fc22 = nn.Linear(100, self.hidden_dim)

    def _init_decoder(self):
        self.decoder: dict[str, BayesVAELinearParams] = OrderedDict()
        self.decoder["fc3"] = BayesVAELinearParams(
            in_features=self.hidden_dim, out_features=400
        )
        self.decoder["fc4"] = BayesVAELinearParams(in_features=400, out_features=784)

    def _init_posterior_theta(self):
        """Initializes the parameters for variational posterior for theta."""
        self.total_params_theta = sum(
            [layer.total_params() for _, layer in self.decoder.items()]
        )

    def draw_posterior_theta(self) -> torch.Tensor:
        theta = self.reparameterize(self.mu_theta, self.logvar_theta)
        return theta

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Forward pass of the VAE.

        Args:
            x: Image tensor of dim. (B, W, H)

        Returns:
            (VAEOutput):
                decoded image, encoder mean, encoder log(variance).
        """
        z, mu, logvar = self.sample(x.view(-1, 784))
        return self.decode(z), mu, logvar

    def update_decoder_params(self, theta: torch.Tensor):
        next_param_idx = 0
        for layer in self.decoder.values():
            next_param_idx = layer.unpack(next_param_idx, theta)
        assert next_param_idx == self.total_params_theta

    def sample(self, x: torch.Tensor):
        """Samples from the encoder distribution.

        Returns:
            sample, encoder mean, encoder log(variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

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
        theta = self.draw_posterior_theta()
        self.update_decoder_params(theta)
        h3 = F.relu(self.decoder["fc3"](z))
        return torch.sigmoid(self.decoder["fc4"](h3))


# --------------------- Convolutional VAE --------------------- #


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
                stride=(1, 1),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                stride=(1, 1),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.Flatten(),
        )

    def _encode_cv_dim(self):
        """Computes the dimension of input img. after encoder layer."""
        # Dimension of image after applying first convlayer
        dim_cv1 = model_utils.compute_outputdim_cv(I=28, F=3, P=1, S=1)
        # Dimension of image after applying second convlayer
        dim_cv2 = model_utils.compute_outputdim_cv(
            I=dim_cv1, F=3, P=1, S=2
        )
        dim_cv3 = model_utils.compute_outputdim_cv(
            I=dim_cv2, F=3, P=1, S=2
        )
        self.img_dim_encoder = model_utils.compute_outputdim_cv(
            I=dim_cv3, F=3, P=1, S=1
        )
        self.total_dim_encoder = 64 * (self.img_dim_encoder**2)

    def _init_decoder(self):
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.total_dim_encoder),
            nn.Unflatten(
                dim=1, unflattened_size=[64, self.img_dim_encoder, self.img_dim_encoder]
            ),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(1, 1),
                kernel_size=(3, 3),
                padding=1,
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
        return self.decode(Z), mu, logvar

    def sample(self, x: torch.Tensor):
        """Samples from the encoder distribution.

        Returns:
            sample, encoder mean, encoder log(variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, X: torch.Tensor) -> EncoderOutput:
        X = self.encoder(X)
        mu, logvar = self.fc_mean(X), self.fc_logvar(X)
        return mu, logvar

    def decode(self, Z: torch.Tensor):
        return self.decoder(Z).view(-1, 784)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# --------------------- Loss functions --------------------- #


class VAELoss:
    """Class to construct loss function for VAE."""

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

    def standard_loss(
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

    def bayesian_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu_encoder: torch.Tensor,
        logvar_encoder: torch.Tensor,
        mu_theta: torch.Tensor,
        logvar_theta: torch.Tensor,
    ):
        """Loss function for Bayesian VAE.

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
        KLD_encoder = -0.5 * torch.sum(
            1 + logvar_encoder - mu_encoder.pow(2) - logvar_encoder.exp()
        )
        KLD_theta = -0.5 * torch.sum(
            1 + logvar_theta - mu_theta.pow(2) - logvar_theta.exp()
        )
        return logpx_loss + KLD_encoder + KLD_theta


# --------------------- Pytorch Lightning --------------------- #


class VAE(Protocol):
    """Protocol for a VAE"""

    def __call__(self, X: torch.Tensor) -> VAEOutput:
        ...

    def sample(self, x: torch.Tensor) -> VAEOutput:
        ...

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Encoder(Protocol):
    """Protocol for an encoder"""

    def encode(self, X: torch.Tensor) -> EncoderOutput:
        ...


class LitVAE(pl.LightningModule):
    def __init__(self, vae: VAE, logpx_loss: str = "bce"):
        super().__init__()
        self.vae = vae
        self.loss_fn = VAELoss(logpx_loss=logpx_loss).standard_loss

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


class LitBayesVAE(pl.LightningModule):
    def __init__(self, vae: BayesVAE, logpx_loss: str = "bce"):
        super().__init__()
        self.vae = vae
        self.loss_fn = VAELoss(logpx_loss=logpx_loss).bayesian_loss

    def training_step(self, batch, _):
        return self.inner_step(batch, step_name="train", prog_bar=True)

    def validation_step(self, batch, _):
        return self.inner_step(batch, step_name="val")

    def inner_step(self, batch, step_name: str, **kwargs):
        """Inner step of both train and validation steps"""
        x, _ = batch  # Don't use target labels
        recon_batch, mu_encoder, logvar_encoder = self.vae(x)
        loss = self.loss_fn(
            recon_x=recon_batch,
            x=x,
            mu_encoder=mu_encoder,
            logvar_encoder=logvar_encoder,
            mu_theta=self.vae.mu_theta,
            logvar_theta=self.vae.logvar_theta,
        )
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

    def __init__(self, save: bool = False, save_suffix: Optional[str] = None):
        super().__init__()
        self._samples = []
        self.save = save
        self.save_suffix = save_suffix
        self.epoch = 0
        self.samples: torch.Tensor

    def on_train_epoch_end(self, _, pl_module):
        with torch.no_grad():
            pl_module.eval()
            sample = torch.randn(64, pl_module.vae.hidden_dim).to(pl_module.device)
            sample = pl_module.vae.decode(sample)
            self._samples.append(sample.view(64, 28, 28))
            pl_module.train()
            if self.save:
                name = f"sample_{self.epoch}_{self.save_suffix}.png"
                save_image(
                    sample.view(64, 1, 28, 28),
                    fp=str(pmldiku.FP_FIGS / name)
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


# --------------------- Marginal likelihood estimation --------------------- #


class MarginalLogLikVAE:
    """
    Class to handle data and methods related to marginal
    loglikehood estimation for variational autoencoders.
    """

    def __init__(
        self,
        val_loader: torch.utils.data.DataLoader,
        model: LitVAE,
        latent_dim: int = 2,
        L: int = 5,
    ):
        self.val_loader = val_loader
        self.model = model
        self.latent_dim = latent_dim
        self.L = L
        self._construct_latent_dist()
        self._init_sample()

    def _construct_latent_dist(self):
        """Construct latent dist. used for all samples"""
        zero_vec = torch.zeros(self.latent_dim)
        self.mvn_latent_prior = tdist.MultivariateNormal(
            loc=zero_vec, covariance_matrix=torch.eye(self.latent_dim)
        )

    def _init_sample(self):
        """Initializes tensor to store log p(x_i) for each sample x_i"""
        self.N_val = len(self.val_loader.dataset)  # type: ignore
        self.logpx_all = torch.zeros(self.N_val)

    def estimate(self):
        counter = 0
        for X_b, _ in tqdm.tqdm(self.val_loader):
            logpx_batch = self.estimate_batch(X_b)
            b_shape = X_b.shape[0]
            self.logpx_all[counter : counter + b_shape] = logpx_batch
            counter += b_shape
        logpx = -np.log(self.N_val) + torch.logsumexp(self.logpx_all, dim=0)
        return logpx

    def estimate_batch(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        logpx_all = torch.zeros(B)
        for i in range(B):
            x = X[i].unsqueeze(0)
            logpx = self.estimate_single(x)
            logpx_all[i] = logpx
        return logpx_all

    def estimate_single(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates log p(x) for a given image tensor x.

        Note:
            To compute the log of an average of a sum of exponentials
            we use the logsumexp-trick.
        """
        samples = torch.zeros(self.L)
        for j in range(self.L):
            z, mean, logvar = self.model.vae.sample(x)
            encoder_var = torch.exp(logvar / 2).squeeze().diag()
            encoder_mean = mean.squeeze()
            mvn_encoder = tdist.MultivariateNormal(
                loc=encoder_mean, covariance_matrix=encoder_var
            )
            logqzgx = mvn_encoder.log_prob(z)
            logpz = self.mvn_latent_prior.log_prob(z)
            recon_x = self.model.vae.decode(z)
            logpxgz = -self.model.loss_fn.logpx_loss_fn(recon_x, x.view(-1, 784))
            samples[j] = logpxgz + logpz - logqzgx
        logpx = -np.log(self.L) + torch.logsumexp(samples, dim=0)
        return logpx


# --------------------- Other --------------------- #


class BayesVAEConv2dTransposeParams:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        kernel_size: tuple[int, int],
        padding: int
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.k1, self.k2 = self.kernel_size
        self.padding = padding
        self.weights = torch.empty(self.in_channels, self.out_channels, self.k1 , self.k2)
        self.bias = torch.empty(self.out_channels)

    def __call__(self, input: torch.Tensor):
        return F.conv_transpose2d(input, weight=self.weights, bias=self.bias, stride=self.stride, padding=self.padding)

    def unpack(self, start_idx: int, theta: torch.Tensor) -> int:
        num_weight_params = self.num_weight_params()
        num_bias_params = self.num_bias_params()
        weights_idx = start_idx + num_weight_params
        bias_idx = weights_idx + num_bias_params
        weights = theta[start_idx:weights_idx]
        bias = theta[weights_idx:bias_idx]
        self.update_params(weights, bias)
        return bias_idx

    def update_params(self, weights: torch.Tensor, bias: torch.Tensor):
        """Updates the parameters with the slice from the drawn theta"""
        self.weights = weights.view((self.in_channels, self.out_channels, self.k1 , self.k2))
        self.bias = bias

    def total_params(self):
        return self.num_weight_params() + self.num_bias_params()

    def num_weight_params(self):
        return self.in_channels * self.out_channels * self.k1 * self.k2

    def num_bias_params(self):
        return self.out_channels


class BayesCVAE(pl.LightningModule):
    def __init__(self, hidden_dim: int = 2):
        super(BayesCVAE, self).__init__()
        self.hidden_dim = hidden_dim
        # Init network
        self._init_encoder()
        self._encode_cv_dim()
        self.fc_mean = nn.Linear(self.total_dim_encoder, hidden_dim)
        self.fc_logvar = nn.Linear(self.total_dim_encoder, hidden_dim)
        self._init_decoder()

        self._init_posterior_theta()
        # pytorch_lightning won't capture the params if we
        # intialize them inside `_init_posterior_theta`
        self.mu_theta = nn.Parameter(torch.randn(self.total_params_theta))
        self.logvar_theta = nn.Parameter(torch.randn(self.total_params_theta))

    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                stride=(1, 1),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                stride=(1, 1),
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.Flatten(),
        )

    def _encode_cv_dim(self):
        """Computes the dimension of input img. after encoder layer."""
        # Dimension of image after applying first convlayer
        dim_cv1 = model_utils.compute_outputdim_cv(I=28, F=3, P=1, S=1)
        # Dimension of image after applying second convlayer
        dim_cv2 = model_utils.compute_outputdim_cv(
            I=dim_cv1, F=3, P=1, S=2
        )
        dim_cv3 = model_utils.compute_outputdim_cv(
            I=dim_cv2, F=3, P=1, S=2
        )
        self.img_dim_encoder = model_utils.compute_outputdim_cv(
            I=dim_cv3, F=3, P=1, S=1
        )
        self.total_dim_encoder = 64 * (self.img_dim_encoder**2)

    def _init_decoder(self):
        self.decoder: dict[str, BayesVAELinearParams | BayesVAEConv2dTransposeParams] = OrderedDict()
        self.decoder["f1"] = BayesVAELinearParams(self.hidden_dim, self.total_dim_encoder)
        self.unflattenf1 = nn.Unflatten(
            dim=1, unflattened_size=[64, self.img_dim_encoder, self.img_dim_encoder]
        )
        self.decoder["f2t2d"] = BayesVAEConv2dTransposeParams(
            in_channels=64,
            out_channels=64,
            stride=(1, 1),
            kernel_size=(3, 3),
            padding=1,
        )
        self.decoder["f3t2d"] = BayesVAEConv2dTransposeParams(
            in_channels=64,
            out_channels=64,
            stride=(2, 2),
            kernel_size=(3, 3),
            padding=1,
        )
        self.decoder["f4t2d"] = BayesVAEConv2dTransposeParams(
            in_channels=64,
            out_channels=32,
            stride=(2, 2),
            kernel_size=(3, 3),
            padding=1,
        )
        self.decoder["f5t2d"] = BayesVAEConv2dTransposeParams(
            in_channels=32,
            out_channels=1,
            stride=(1, 1),
            kernel_size=(4, 4),
            padding=0,
        )

    def _init_posterior_theta(self):
        """Initializes the parameters for variational posterior for theta."""
        self.total_params_theta = sum(
            [layer.total_params() for _, layer in self.decoder.items()]
        )

    def draw_posterior_theta(self) -> torch.Tensor:
        theta = self.reparameterize(self.mu_theta, self.logvar_theta)
        return theta

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Forward pass of the VAE.

        Args:
            x: Image tensor of dim. (B, W, H)

        Returns:
            (VAEOutput):
                decoded image, encoder mean, encoder log(variance).
        """
        z, mu, logvar = self.sample(x)
        return self.decode(z), mu, logvar

    def update_decoder_params(self, theta: torch.Tensor):
        next_param_idx = 0
        for layer in self.decoder.values():
            next_param_idx = layer.unpack(next_param_idx, theta)
        assert next_param_idx == self.total_params_theta

    def sample(self, x: torch.Tensor):
        """Samples from the encoder distribution.

        Returns:
            sample, encoder mean, encoder log(variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, X: torch.Tensor) -> EncoderOutput:
        """Encodes the input image tensor.

        The image is encoded into the mean and variance
        of the encoder distribution q(z | x).

        Args:
            X: (B, 784) dimensional image tensor

        Returns:
            (EncoderOutput):
                Mean and variance of encoder distribution.
        """
        X = self.encoder(X)
        mu, logvar = self.fc_mean(X), self.fc_logvar(X)
        return mu, logvar

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
        theta = self.draw_posterior_theta()
        self.update_decoder_params(theta)
        h1 = self.unflattenf1(self.decoder["f1"](z))
        h2 = F.leaky_relu(self.decoder["f2t2d"](h1))
        h3 = F.leaky_relu(self.decoder["f3t2d"](h2))
        h4 = F.leaky_relu(self.decoder["f4t2d"](h3))
        return torch.sigmoid(self.decoder["f5t2d"](h4)).view(-1, 784)
