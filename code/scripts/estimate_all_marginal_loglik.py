from pathlib import Path

import torch

import pmldiku
from pmldiku import data, vae

# --------------------- Globals --------------------- #

torch.manual_seed(1)
CUDA = False
BATCH_SIZE = 128
LOGPX_LOSS = "bce"
DEVICE_NAME = "cuda" if CUDA else "cpu"
DEVICE = torch.device(DEVICE_NAME)
kwargs = {"num_workers": 4, "pin_memory": DEVICE}
val_loader = data.load_mnist(train=False).setup_data_loader(
    batch_size=BATCH_SIZE, **kwargs
)

MODEL_CHECKPOINTS = [
    "mnist-bayes-bce-h2-epoch=136-val_loss=53232.71.ckpt",
    "mnist-bayes-cb-h2-epoch=193-val_loss=-373011.47.ckpt",
    "mnist-bayes-mse-h2-epoch=59-val_loss=26809.54.ckpt",
    "mnist-conv-bce-h2-epoch=20-val_loss=38819.56.ckpt",
    "mnist-conv-cb-h2-epoch=28-val_loss=-400214.72.ckpt",
    "mnist-conv-mse-h2-epoch=45-val_loss=9583.45.ckpt",
    "mnist-vanilla-bce-h2-epoch=43-val_loss=36115.11.ckpt",
    "mnist-vanilla-cb-h2-epoch=51-val_loss=-407834.12.ckpt",
    "mnist-vanilla-mse-h2-epoch=34-val_loss=8956.06-v3.ckpt",
]

FP_LOGLIK = pmldiku.FP_OUTPUT / "loglik"
Path.mkdir(FP_LOGLIK, exist_ok=True)

# --------------------- Estimate marginal loglik --------------------- #


def estimate_marginal_loglik(model_type: str, loss_fn_name: str, hidden_dim: int = 2):
    # Set seed
    torch.manual_seed(1)

    if loss_fn_name not in ["mse", "bce", "cb"]:
        raise ValueError(
            f"Only MSE, BCE and CB loss functions implemented; got {loss_fn_name}"
        )

    model_file_name = f"mnist-{model_type}-{loss_fn_name}"
    path, = [file for file in MODEL_CHECKPOINTS if file.startswith(model_file_name)]
    assert isinstance(path, str)
    path = pmldiku.FP_MODELS / path

    match model_type:
        case "bayes":
            base_vae = vae.BayesVAE(hidden_dim=hidden_dim)
            model = vae.LitBayesVAE.load_from_checkpoint(checkpoint_path=path, vae=base_vae, logpx_loss=loss_fn_name)
        case "conv":
            base_vae = vae.CVAE(hidden_dim=hidden_dim)
            model = vae.LitVAE.load_from_checkpoint(checkpoint_path=path, vae=base_vae, logpx_loss=loss_fn_name)
        case "vanilla":
            base_vae = vae.BaseVAE(hidden_dim=hidden_dim)
            model = vae.LitVAE.load_from_checkpoint(checkpoint_path=path, vae=base_vae, logpx_loss=loss_fn_name)
        case _:
            raise ValueError(
                f"Got wrong {model_type}; only bayes, conv and vanilla implemented"
            )
    print(f"Estimating marginal loglik for {model_file_name}")
    mloglik_vae = vae.MarginalLogLikVAE(val_loader, model, device=DEVICE, latent_dim=2, L=5)
    logpx_vae = mloglik_vae.estimate()
    with open(FP_LOGLIK / "marginalloglikvalues.txt", "a") as file:
        file.write(f"{model_file_name}, {logpx_vae}")
    return logpx_vae


# --------------------- Main --------------------- #


def main():
    for loss_fn_name in ["mse", "bce", "cb"]:
        for model_type in ["vanilla", "conv", "bayes"]:
            estimate_marginal_loglik(
                model_type=model_type, loss_fn_name=loss_fn_name, hidden_dim=2
            )


if __name__ == "__main__":
    main()
