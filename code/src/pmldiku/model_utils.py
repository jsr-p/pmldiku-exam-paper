import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import pmldiku


def plot_img(img: np.ndarray):
    plt.imshow(img, cmap="gray")
    plt.axis("off")


def plot_loss(loss: np.ndarray, **kwargs):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ts = np.arange(1, loss.shape[0] + 1)
    ax.plot(ts, loss)
    ax.set(title="Loss", xlabel="epoch", ylabel="loss", **kwargs)


def plot_image_reconstruction(
    images: np.ndarray,
    num_cols: int = 3,
    slim: int = 30,
    start: int = 0,
    multi_title=True,
    **kwargs,
):
    """Plots reconstructed images in a grid.

    Args:
        images: array of dim (N, X, Y) where N is number of imgs.
        num_cols: number of columns in img.
        slim: spacing between imgs.
        start: epoch of the first image in grid
    """
    N = images.shape[0]
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_rows, 4 * num_cols)
    )

    for i, ax in zip(range(start, N + start), axes.flatten()):  # type: ignore

        ax.imshow(images[i - start], cmap="gray")
        if multi_title:
            ax.set(title=f"{kwargs['title']} {i + 1}")
        else:
            if i == 0:
                ax.set(title=f"{kwargs['title']}")
        ax.axis("off")
    fig.tight_layout(w_pad=-slim)
    return fig


def plot_encoded(X: np.ndarray, labels: np.ndarray, **kwargs):
    scatter_x = X[:, 0]
    scatter_y = X[:, 1]
    group = labels

    cdict = {
        0: "black",
        1: "red",
        2: "blue",
        3: "green",
        4: "brown",
        5: "orange",
        6: "yellow",
        7: "magenta",
        8: "cyan",
        9: "purple",
    }

    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=0.8)
    ax.legend(markerscale=10, title="Image labels", bbox_to_anchor=(1, 1))
    if not kwargs:
        kwargs = {"title": "Encoded images"}
    ax.set(**kwargs)
    fig.tight_layout()
    plt.show()


def construct_gauss_grid(M: int):
    x = torch.linspace(start=0, end=1, steps=M + 2)
    x = x[1:-1]
    mesh_x, mesh_y = torch.meshgrid(x, x, indexing="ij")
    gauss = torch.distributions.normal.Normal(loc=0, scale=1)
    gauss_x = gauss.icdf(mesh_x).view(-1, 1)
    gauss_y = gauss.icdf(mesh_y).view(-1, 1)
    gauss_vals = torch.column_stack((gauss_x, gauss_y))
    return gauss_vals


def plot_gauss_grid_imgs(decoded_imgs: np.ndarray):
    # Plot
    M = int(np.sqrt(decoded_imgs.shape[0]))
    fig, axes = plt.subplots(nrows=M, ncols=M, figsize=(8, 8), sharex=True, sharey=True)
    for i, ax in zip(range(M * M), axes.flatten()):  # type: ignore
        img = decoded_imgs[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    slim = 0.6
    fig.tight_layout(pad=-slim, w_pad=-slim, h_pad=-slim)
    plt.show()


def compute_outputdim_cv(I: int, F: int, P: int, S: int):
    """Computes the output dimension of an image after convolution.

    See:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        I: Length of input volumne (img. dimension)
        F: Length of filter
        P: Padding length
        S: Stride length

    Return:
        (int): output img. dimension
    """
    O = math.floor((I + 2 * P - F) / S + 1)
    return O


def filesize(fp: Path):
    size_mb = fp.stat().st_size / 10**6
    return size_mb


@dataclass
class TrainedModels:
    models: dict[str, Path]

    def __post_init__(self):
        self.models_size = {fp.name: filesize(fp) for fp in self.models.values()}

    def print_overview(self):
        for model in self.models:
            model_size = self.models_size[model]
            print(f"Model {model}; size {model_size:.2f} mb.")


def show_trained_models() -> TrainedModels:
    models = {fp.name: fp for fp in pmldiku.FP_MODELS.glob("*")}
    trained_models = TrainedModels(models=models)
    trained_models.print_overview()
    return trained_models


def save_image_tensor(
    images: torch.Tensor,
    path: Path,
    fname: str,
    strict: bool = True,
) -> None:
    """Saves images
    images: Tensor with images
    strict: ensures tensor is of shape (10000, 1, 28, 28
    """
    if strict:
        assert images.shape == (10000, 1, 28, 28)
    torch.save(images, path / Path(fname))


def load_image_tensor(path: Path, fname: str) -> torch.Tensor:
    return torch.load(path / Path(fname))


if __name__ == "__main__":
    print(pmldiku.FP_MODELS)
