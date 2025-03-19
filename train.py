import os
import matplotlib
import matplotlib.pyplot as plt
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ddpm.ddpm import Diffusion
from ddpm.cifar10 import CIFAR10DataModule

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "ckpt/")


class DDPM(pl.LightningModule):
    def __init__(self, T, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = Diffusion(T)
        self.criteria = nn.MSELoss(reduction="sum")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        B = imgs.shape[0]

        t = torch.randint(0, self.hparams.T, (B, ), device=self.device).long()
        x_t, noise = self.model.sample(imgs, t)

        preds = self.model(x_t, t)
        loss = self.criteria(preds, noise)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")



def train_model(**kwargs):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "DDPM"),
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT/vit.ckpt")
    dm = CIFAR10DataModule()
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DDPM.load_from_checkpoint(pretrained_filename)
    else:
        model = DDPM(T=kwargs["T"], lr=kwargs["lr"])
        trainer.fit(model, datamodule=dm)
        model = DDPM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    test_result = trainer.test(model, datamodule=dm, verbose=False)
    val_result = trainer.test(model, datamodule=dm, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

model, results = train_model(
    T = 1000,
    lr=3e-4,
)
print("ddpm results", results)
