from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torchvision.transforms import Compose, Lambda, ToTensor

import pmldiku
from pmldiku import data, diffusion, diffusion_utils, model_utils, output_utils


# --------------------- Globals --------------------- #

cuda = True
batch_size = 128
epochs = 20
device_name = "cuda" if cuda else "cpu"
unet = True

device = torch.device(device_name)
kwargs = {"num_workers": 4, "pin_memory": device}

n_steps, min_beta, max_beta = 1000, 10**-4, 0.02  # Originally used by the authors
config = diffusion.ParamConfig(n_steps, min_beta, max_beta)

code_path = pmldiku.FP_PROJ
mu_, std_ = 0.1319, 0.3094  # empircal mean and std of the full sample

transform = Compose([ToTensor(), Lambda(lambda x: (x - mu_) * (1 / std_))])

loader = data.load_mnist(train=True, trans=transform).setup_data_loader(
    batch_size=batch_size, **kwargs
)

tensor_path = pmldiku.FP_OUTPUT / "image-tensors"
Path.mkdir(tensor_path, exist_ok=True)


# --------------------- Run model --------------------- #


def run_diffusion_model(unet: bool = False):
    torch.manual_seed(1)
    if unet:
        network = diffusion.UNet
        model_name = "UNet"
    else:
        network = diffusion.ConvNet
        model_name = "ConvNet"

    model = diffusion.LightningDiffusion(config=config, network=network, verbose=True)
    model_cp_name = f"mnist-diffusion-{model_name}"
    checkpoint_fname = model_cp_name + "{epoch:02d}-{val_loss:.2f}"
    print(f"Logging model with name {model_cp_name}")
    cb_model_checkpoint, cb_early_stopping, cb_loss, cb_lr_monitor = (
        ModelCheckpoint(
            dirpath=pmldiku.FP_MODELS,
            filename=checkpoint_fname,
        ),
        EarlyStopping("mse_metric"),
        diffusion_utils.DiffusionLossCallback(),
        LearningRateMonitor("step"),
    )

    callbacks = [
        cb_model_checkpoint,
        cb_early_stopping,
        cb_loss,
        cb_lr_monitor,
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator=device_name,
        callbacks=callbacks,
        auto_lr_find=True,
    )
    trainer.fit(model=model, train_dataloaders=loader)

    losses = np.array(cb_loss.train_loss)
    model_utils.plot_loss(losses)

    ims = model.to(device).generate(10000)
    standardized_ims = ((ims - ims.mean()) / ims.std()).cpu()
    model_utils.save_image_tensor(
        standardized_ims, tensor_path, f"{model_name}_diffusion.pkl"
    )
    fig = model_utils.plot_image_reconstruction(
        ims[0:9].cpu().detach().numpy().reshape(9, 28, 28),
        num_cols=3,
        slim=0,
        title=f"{model_name} diffusion",
        multi_title=False,
    )
    figs_path = pmldiku.FP_FIGS
    output_utils.save_fig(fig, figs_path, f"{model_name}-reconstruction")


def main():
    run_diffusion_model(unet=True)
    run_diffusion_model(unet=False)


if __name__ == "__main__":
    main()
