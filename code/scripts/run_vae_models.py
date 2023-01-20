from pathlib import Path

import torch
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np

import pmldiku
from pmldiku import data, vae, model_utils, output_utils


# --------------------- Globals --------------------- #


CUDA = True
BATCH_SIZE = 256
DEVICE_NAME = "cuda" if CUDA else "cpu"
torch.manual_seed(1)

DEVICE = torch.device(DEVICE_NAME)
kwargs = {"num_workers": 4, "pin_memory": DEVICE}

train_loader = data.load_mnist(train=True).setup_data_loader(
    batch_size=BATCH_SIZE, **kwargs
)
val_loader = data.load_mnist(train=False).setup_data_loader(
    batch_size=BATCH_SIZE, **kwargs
)


fp_loss_tensor_path = pmldiku.FP_OUTPUT / "loss_tensors"
fp_tensor_path = pmldiku.FP_OUTPUT / "image-tensors"
for fp in [fp_loss_tensor_path, fp_tensor_path]:
    Path.mkdir(fp, exist_ok=True)

# --------------------- Run model --------------------- #


def run_vae_model(model_type: str, loss_fn_name: str, hidden_dim: int = 2):
    # Set seed
    torch.manual_seed(1)

    if loss_fn_name not in ["mse", "bce", "cb"]:
        raise ValueError(
            f"Only MSE, BCE and CB loss functions implemented; got {loss_fn_name}"
        )

    match model_type:
        case "bayes":
            bayes_vae = vae.BayesVAE(hidden_dim=hidden_dim)
            model = vae.LitBayesVAE(vae=bayes_vae, logpx_loss=loss_fn_name)
        case "conv":
            base_vae = vae.CVAE(hidden_dim=2)
            model = vae.LitVAE(vae=base_vae, logpx_loss=loss_fn_name)
        case "vanilla":
            base_vae = vae.BaseVAE(hidden_dim=hidden_dim)
            model = vae.LitVAE(vae=base_vae, logpx_loss=loss_fn_name)
        case _:
            raise ValueError(
                f"Got wrong {model_type}; only bayes, conv and vanilla implemented"
            )

    # Callbacks
    model_file_name = f"mnist-{model_type}-{loss_fn_name}-h{hidden_dim}-"
    checkpoint_fname = model_file_name + "{epoch:02d}-{val_loss:.2f}"
    loss_callback = vae.VAELossCallback()
    reconstruct_cb = vae.VAEImageReconstructionCallback()
    cb_checkpoint = ModelCheckpoint(
        dirpath=pmldiku.FP_MODELS,
        filename=checkpoint_fname,
    )
    cb_earlystopping = EarlyStopping(monitor="val_loss", mode="min")
    callbacks = [loss_callback, reconstruct_cb, cb_checkpoint, cb_earlystopping]

    # Trainer
    print(f"Estimating model {model_file_name}...")
    trainer = pl.Trainer(
        max_epochs=-1, devices=1, accelerator=DEVICE_NAME, callbacks=callbacks
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Finished estimating model {model_file_name}...")

    # Loss
    with open(fp_loss_tensor_path / f"{model_file_name}-loss.npy", "wb") as file:
        losses = np.array(loss_callback.val_loss)
        model_utils.plot_loss(losses)
        np.save(file, losses)

    # Reconstructed images in last epoch; for Freschet comparison
    latent_draws = torch.randn((10_000, 2))
    decoded_imgs = model.vae.decode(latent_draws).detach().numpy()
    model_utils.save_image_tensor(
        decoded_imgs, fp_tensor_path, f"{model_file_name}_diffusion.pkl"
    )
    fig = model_utils.plot_image_reconstruction(
        decoded_imgs[:9].reshape(9, 28, 28),
        num_cols=3,
        slim=0,
        title=f"{model_file_name}-imgtensors",
        multi_title=False,
    )
    figs_path = pmldiku.FP_FIGS
    output_utils.save_fig(fig, figs_path, f"{model_file_name}-reconstruction")

    # Image reconstructed epoch after epoch
    images = reconstruct_cb.samples[-16:, 0, :, :].cpu().numpy()
    start = cb_earlystopping.stopped_epoch - 16
    fig = model_utils.plot_image_reconstruction(
        images, num_cols=4, slim=20, start=start
    )
    fig.tight_layout()
    fname = pmldiku.FP_FIGS / f"{model_file_name}-image-epochs-reconstruction"
    fig.savefig(str(fname))

    # Encoded means
    means_plot, labels = vae.encode_means(model.vae, val_loader)
    plot_args = dict(title="Encoded means", xlabel=r"$\mu_1$", ylabel=r"$\mu_2$")
    fig = model_utils.plot_encoded(means_plot, labels, **plot_args)
    fname = pmldiku.FP_FIGS / f"{model_file_name}-encoded_means"
    fig.savefig(str(fname))

    # Gauss grid
    gauss_vals = model_utils.construct_gauss_grid(M=12)
    decoded_imgs = model.vae.decode(gauss_vals).detach().numpy()
    fig = model_utils.plot_gauss_grid_imgs(decoded_imgs)
    fname = pmldiku.FP_FIGS / f"{model_file_name}-gauss-grid"
    fig.savefig(str(fname))

    print(f"Finished creating all output for {model_file_name}")


# --------------------- Main --------------------- #


def main():
    for loss_fn_name in ["mse", "bce", "cb"]:
        for model_type in ["vanilla", "conv", "bayes"]:
            run_vae_model(
                model_type=model_type, loss_fn_name=loss_fn_name, hidden_dim=2
            )


if __name__ == "__main__":
    main()
