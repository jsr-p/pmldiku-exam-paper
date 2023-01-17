import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch


def plot_img(img: np.ndarray):
    plt.imshow(img, cmap="gray")
    plt.axis("off")


def plot_loss(loss: np.ndarray, **kwargs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ts = np.arange(1, loss.shape[0] + 1)
    ax.plot(ts, loss)
    ax.set(title="Loss", xlabel="epoch", ylabel="loss", **kwargs)


def plot_image_reconstruction(images: np.ndarray, num_cols: int = 3, slim: int = 30):
    """Plots reconstructed images in a grid.

    Args:
        images: array of dim (N, X, Y) where N is number of imgs.
    """
    N = images.shape[0]
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_rows, 4 * num_cols)
    )
    for i, ax in zip(range(N), axes.flatten()):  # type: ignore
        ax.imshow(images[i], cmap="gray")
        ax.set(title=f"Reconstruction of img. at epoch {i + 1}")
        ax.axis("off")
    fig.tight_layout(w_pad=-slim)


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

    Args:
        I: Length of input volumne (img. dimension)
        F: Length of filter
        P: Padding length
        S: Stride length
    Return:
        (int): output img. dimension
    """
    O = (I - F + 2 * P) // S + 1
    return O
