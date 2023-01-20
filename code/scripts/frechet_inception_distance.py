import torch
import torch.utils.data

import pytorch_lightning as pl


from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

import pmldiku
from pmldiku import data, model_utils, output_utils, fid


torch.manual_seed(1)

cuda = True
batch_size = 128
epochs = 200  # 200
device_name = "cuda" if cuda else "cpu"

device = torch.device(device_name)
kwargs = {"num_workers": 4, "pin_memory": device}

code_path = pmldiku.FP_PROJ

train_loader = data.load_mnist(train=True).setup_data_loader(
    batch_size=batch_size, **kwargs
)
validation_loader = data.load_mnist(train=False).setup_data_loader(
    batch_size=batch_size, **kwargs
)
tensor_path = pmldiku.FP_OUTPUT / "image-tensors"


def fit_classifier():
    if (file := (pmldiku.FP_MODELS / "frechet-classifier.ckpt")).exists():
        classifier = fid.MNISTClassifier.load_from_checkpoint(file)
        return classifier

    classifier: fid.MNISTClassifier = fid.MNISTClassifier()

    (
        cb_model_checkpoint,
        cb_early_stopping,
        cb_lr_monitor,
        cb_class_loss,
    ) = (
        ModelCheckpoint(
            dirpath=pmldiku.FP_MODELS,
            filename="frechet-classifier",
        ),
        EarlyStopping("val_mse"),
        LearningRateMonitor("step"),
        fid.ClassifierLossCallback(),
    )

    callbacks = [
        cb_model_checkpoint,
        cb_early_stopping,
        cb_lr_monitor,
        cb_class_loss,
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator=device_name,
        auto_lr_find=True,
        callbacks=callbacks,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    return classifier


def main():
    classifier = fit_classifier()
    batch_size = 10000
    validation_loader = data.load_mnist(train=False).setup_data_loader(
        batch_size=batch_size, **kwargs
    )
    batch = next(iter(validation_loader))
    X_true, y_true = batch
    y_hat = classifier.model(X_true)

    # performance in validation set
    n_correct = fid.avg_correct_label(y_hat.detach().argmax(1), y_true)

    # # Calculating FIDS
    FSO = output_utils.FIDScoreOutput(n_decimals=2)

    # bernoulli with p = 0.9 reasonable baseline
    X_baseline = torch.bernoulli(torch.ones(10000, 1, 28, 28) * 0.09)
    model_utils.save_image_tensor(X_baseline, tensor_path, "baseline.pkl")

    frechet_inception_distance = fid.FID(classifier)
    FSO = output_utils.FIDScoreOutput(n_decimals=2)

    # Load image tensors of dimension: (10, 1, 28, 28)

    X_baseline = model_utils.load_image_tensor(tensor_path, "baseline.pkl")
    X_vae = model_utils.load_image_tensor(tensor_path, "mnist-vanilla-cb-h2-_diffusion.pkl")
    X_vbae = model_utils.load_image_tensor(tensor_path, "mnist-bayes-cb-h2-_diffusion.pkl")
    X_cvea = model_utils.load_image_tensor(tensor_path, "mnist-conv-cb-h2-_diffusion.pkl")
    X_convbce = model_utils.load_image_tensor(tensor_path, "mnist-conv-bce-h2-_diffusion.pkl")
    X_unet_diffusion: torch.Tensor = model_utils.load_image_tensor(
        tensor_path, "UNet_diffusion.pkl"
    )
    X_conv_diffusion: torch.Tensor = model_utils.load_image_tensor(
        tensor_path, 'ConvNet_diffusion.pkl'
    )

    tensors = [X_vae, X_vbae, X_cvea, X_convbce, X_unet_diffusion, X_conv_diffusion]
    names = ["vae", "vbae", "cvea", "convbce", "unet_diffusion", "conv_diffusion"]

    for name, tensor in zip(names, tensors):
        score, _ = frechet_inception_distance.calculate_fid(
            X_true, tensor
        )
        FSO.add(name, score.item())

    # Generate table of results
    table = FSO.generate_table()
    with open(pmldiku.FP_PROJ / "tables" / "fso_comparison.tex", "w") as table_file:
        table_file.write(table)

    print(FSO.memory)


if __name__ == "__main__":
    main()
