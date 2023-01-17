from dataclasses import dataclass
from typing import Protocol, TypeAlias

import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn, optim

# --------------------- UNET --------------------- #


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


class MyBlock(nn.Module):
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
    ):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10),
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20),
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1), nn.SiLU(), nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40),
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1),
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20),
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10),
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False),
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

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )


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
    def __init__(self, config: DiffusionParams, verbose: bool = False):
        super(LightningDiffusion, self).__init__()
        self.config = config
        self.verbose = verbose

        self.diffusion_params = DiffusionParams(
            n_steps=config.n_steps, min_beta=config.min_beta, max_beta=config.max_beta
        )
        self.network = MyUNet()
        self.diffusion = Diffusion(network=self.network, params=self.diffusion_params)

    def on_train_start(self):
        print(f"Setting diffusion params to: {self.device}")
        self.diffusion_params.set_params_device(self.device)

    def training_step(self, batch, _):
        return self.inner_step(batch, step_name="training_step")

    def validation_step(self, *_):
        pass

    def inner_step(self, batch, step_name: str):
        x0, _ = batch
        n = len(x0)

        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
        eta = torch.randn_like(x0).to(self.device)
        t = torch.randint(0, self.diffusion_params.n_steps, (n,)).to(self.device)

        # Computing the noisy image based on x0 and the time-step (forward process)
        noisy_imgs = self.diffusion(x0, t, eta)

        # Getting model estimation of noise based on the images and the time-step
        eta_theta = self.diffusion.backward(noisy_imgs, t.reshape(n, -1))

        # Optimizing the MSE between the noise plugged and the predicted noise
        mse = self.diffusion.loss(eta_theta, eta)

        if self.verbose:
            print(f"{step_name}_loss {mse}")
            self.log(f"{step_name}_loss {mse}", mse, prog_bar=True)
        else:
            self.log(f"{step_name}_loss {mse}", mse)
        return mse

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
