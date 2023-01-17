import pytorch_lightning as pl

# callbacks
class DiffusionLossCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.train_loss = []

    def on_train_batch_end(self, trainer, _, outputs, batch, batch_idx):
        self.train_loss.append(trainer.callback_metrics["mse_metric"].item())