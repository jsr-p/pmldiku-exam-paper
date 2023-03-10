from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias
import numpy as np

import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn, optim

# --------------------- UNET --------------------- #

# WE HAVE FOLLOWED THIS BLOGPOST WITH REGARD TO IMPLEMENTATION DETAILS
# ON HOW TO CONSTRUCT A DIFFUSION MODEL AND CREATE A UNET: https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
# THAT IS AN IMPLEMENTATION OF THE ORIGINAL PAPER https://arxiv.org/abs/2006.11239
# 

def sinusoidal_embedding(n: int, d: int) -> torch.Tensor:
    """returns embeddings"""
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


class UNetBlock(nn.Module):
    def __init__(
        self,
        shape,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=None,
        normalize=True,
    ) -> None:
        super(UNetBlock, self).__init__()
        self.ln: nn.LayerNorm = nn.LayerNorm(shape)
        self.conv1: nn.Conv2d = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2: nn.Conv2d = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation: nn.Silu | Any = nn.SiLU() if activation is None else activation
        self.normalize: bool = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_time_embedding(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            UNetBlock((1, 28, 28), 1, 10),
            UNetBlock((10, 28, 28), 10, 10),
            UNetBlock((10, 28, 28), 10, 10),
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_time_embedding(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            UNetBlock((10, 14, 14), 10, 20),
            UNetBlock((20, 14, 14), 20, 20),
            UNetBlock((20, 14, 14), 20, 20),
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_time_embedding(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            UNetBlock((20, 7, 7), 20, 40),
            UNetBlock((40, 7, 7), 40, 40),
            UNetBlock((40, 7, 7), 40, 40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1), nn.SiLU(), nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_time_embedding(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            UNetBlock((40, 3, 3), 40, 20),
            UNetBlock((20, 3, 3), 20, 20),
            UNetBlock((20, 3, 3), 20, 40),
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1),
        )

        self.te4 = self._make_time_embedding(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            UNetBlock((80, 7, 7), 80, 40),
            UNetBlock((40, 7, 7), 40, 20),
            UNetBlock((20, 7, 7), 20, 20),
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_time_embedding(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            UNetBlock((40, 14, 14), 40, 20),
            UNetBlock((20, 14, 14), 20, 10),
            UNetBlock((10, 14, 14), 10, 10),
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_time_embedding(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            UNetBlock((20, 28, 28), 20, 10),
            UNetBlock((10, 28, 28), 10, 10),
            UNetBlock((10, 28, 28), 10, 10, normalize=False),
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(
            self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
        )  # (N, 20, 14, 14)
        out3 = self.b3(
            self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
        )  # (N, 40, 7, 7)

        out_mid = self.b_mid(
            self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
        )  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_time_embedding(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )


# Smaller CONV Network

class ConvNet(nn.Module):
    def __init__(self, n_steps: int=1000, time_emb_dim: int=100) -> None:
        super(ConvNet, self).__init__()

        self.n_steps: int = n_steps
        self.time_emb_dim = time_emb_dim

        self.init_sinusoidal_embed()
        self.init_network()

    def forward(self, x, t) -> torch.Tensor:
        t = self.time_embed(t)
        n = len(x)

        out = self.c_layer_1(x + self.te_1(t).reshape(n, -1, 1, 1))
        out = self.activation(out)
        #out = self.transform_1(out)

        out = self.c_layer_2(out + self.te_2(t).reshape(n, -1, 1, 1))
        out = self.activation(out)
        #out = self.transform_2_1(out)
        #out = self.activation(out)
        #out = self.transform_2_2(out)

        out = self.c_layer_3(out + self.te_3(t).reshape(n, -1, 1, 1))
        #out = self.activation(out)
        #out = self.transform_3(out)

        
        out = self.c_layer_4(out + self.te_4(t).reshape(n, -1, 1, 1))
        #out = self.activation(out)
        out = self.out_layer(out)

        return out

    def init_sinusoidal_embed(self) -> None:
        self.time_embed: nn.Embedding = nn.Embedding(self.n_steps, self.time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(self.n_steps, self.time_emb_dim)
        self.time_embed.requires_grad_(False)
    
    def _make_time_embedding(self, dim_in, dim_out) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )

    def _make_linear_normalize(self, shape) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(shape), nn.Linear(shape[-1], shape[-1])
        )

    def init_network(self) -> None:

        self.activation = nn.ReLU()

        self.te_1 = self._make_time_embedding(self.time_emb_dim, 1)
        self.c_layer_1: nn.Conv2d = nn.Conv2d(
                in_channels=1,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            )
        self.transform_1 = self._make_linear_normalize((32, 14, 14))

        self.te_2: nn.Sequential = self._make_time_embedding(self.time_emb_dim, 32)
        self.c_layer_2: nn.Conv2d = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
            )
        self.transform_2_1: nn.Sequential = self._make_linear_normalize((64, 7, 7))
        self.transform_2_2: nn.Sequential = self._make_linear_normalize((64, 7, 7))

        self.te_3: nn.Sequential = self._make_time_embedding(self.time_emb_dim, 64)
        self.c_layer_3: nn.ConvTranspose2d = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(5, 5),
                padding=1,
            )
        self.transform_3: nn.Sequential = self._make_linear_normalize((32, 15, 15))

        self.te_4: nn.Sequential = self._make_time_embedding(self.time_emb_dim, 32)
        self.c_layer_4: nn.ConvTranspose2d = nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                stride=(2, 2),
                kernel_size=(2, 2),
                padding=1,
            )
            
        self.out_layer = nn.Conv2d(1, 1, 3, 1, 1)


# ---------------------  Model  --------------------- #


@dataclass
class ParamConfig:
    n_steps: int
    min_beta: float
    max_beta: float


class DiffusionParams(pl.LightningModule):
    def __init__(self, n_steps, min_beta, max_beta):
        super(DiffusionParams, self).__init__()
        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.betas = torch.linspace(
            min_beta, max_beta, n_steps
        )  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        )

    def set_params_device(self, device):
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)


class Diffusion(pl.LightningModule):
    def __init__(
        self, network, params: DiffusionParams, image_chw=(1, 28, 28), loss=nn.MSELoss()
    ):
        super(Diffusion, self).__init__()
        self.image_chw = image_chw
        self.loss = loss
        self.network = network
        self.params = params

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, *_ = x0.shape
        a_bar = self.params.alpha_bars[t]
        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1) * x0
            + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        )
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)


DiffusionOutput: TypeAlias = torch.Tensor


class DiffusionModel(Protocol):
    def __call__(self, X: torch.Tensor) -> DiffusionOutput:
        ...


class LightningDiffusion(pl.LightningModule):
    def __init__(self, config: DiffusionParams, network: pl.LightningModule, learning_rate: float=0.001, verbose: bool = False):
        super(LightningDiffusion, self).__init__()
        self.config = config
        self.verbose = verbose
        self.learning_rate = learning_rate

        self.diffusion_params = DiffusionParams(
            n_steps=config.n_steps, min_beta=config.min_beta, max_beta=config.max_beta
        )
        self.network = network()
        self.diffusion = Diffusion(network=self.network, params=self.diffusion_params)

        self.save_hyperparameters()

    def on_train_start(self):
        print(f"Setting diffusion params to: {self.device}")
        self.diffusion_params.set_params_device(self.device)

    def training_step(self, batch, _):
        loss = self.inner_step(batch, step_name="training_step")
        self.log('mse_metric', loss)
        return loss

    def validation_step(self, _):
        pass

    def inner_step(self, batch, step_name: str):
        x0, _ = batch
        n = len(x0)

        # Picking some noise and alpha_bars
        eta = torch.randn_like(x0).to(self.device)
        t = torch.randint(0, self.diffusion_params.n_steps, (n,)).to(self.device)

        # noisy img(forward process)
        noisy_imgs = self.diffusion(x0, t, eta)

        eta_theta = self.diffusion.backward(noisy_imgs, t.reshape(n, -1))

        # MSE between noise and True
        mse = self.diffusion.loss(eta_theta, eta)

        if self.verbose:
            #print(f"{step_name}_loss {mse}")
            self.log(f"{step_name}_loss {mse}", mse, prog_bar=True)
        else:
            self.log(f"{step_name}_loss {mse}", mse)
        return mse

    def training_epoch_end(self, training_step_outputs) -> None:
        if self.verbose:
            last_outputs = training_step_outputs[-50: -1]
            res = np.mean([d['loss'].cpu().numpy() for d in last_outputs])
            print(f"New Epoch. Loss: {res}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate(self, n_samples):

        c, h, w = 1, 28, 28
        print('lars')
        with torch.no_grad():
            x = torch.randn(n_samples, c, h, w).to(self.device)

            for t in list(range(self.diffusion_params.n_steps))[::-1]:
                if t % 50 == 0:
                    print(f"t = {t}")
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device).long()
                
                #print(time_tensor.device, x.device)
                eta_theta = self.diffusion.backward(x, time_tensor)
                alpha_t = self.diffusion_params.alphas[t]
                alpha_t_bar = self.diffusion_params.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                )

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(self.device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = self.diffusion_params.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z
            
            return x
