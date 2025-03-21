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
from tqdm import tqdm
from einops import rearrange, repeat

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "ckpt/")


class DDPMTrainer(pl.LightningModule):
    def __init__(self, T, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = Diffusion(T)

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
        loss = F.mse_loss(preds, noise)

        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss


class DDPMSample(nn.Module):
    def __init__(self, model, T, beta = [0.0001, 0.02]):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        self.register_buffer("one_alpha_t", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff", self.one_alpha_t * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("sigma_t", torch.sqrt(self.beta_t))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = self.one_alpha_t[time_step] * x_t - self.coeff[time_step] * self.model(x_t, t) + self.sigma_t[time_step] * z
        return x_t_minus_one


    @torch.no_grad()
    def forward(self, x_t):
        x = [x_t]
        record = [800, 500, 300, 100, 50, 10, 0]
        for t in reversed(range(self.T)):
            x_t = self.sample_one_step(x_t, t)
            if t in record:
                x.append(torch.clip(x_t, -1.0, 1.0))
        
        return torch.stack(x, dim=1)


def train_model(**kwargs):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "DDPM"),
        max_epochs=20,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "DDPM/ddpm.ckpt")
    dm = CIFAR10DataModule()
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DDPMTrainer.load_from_checkpoint(pretrained_filename)
    else:
        model = DDPMTrainer(T=kwargs["T"], lr=kwargs["lr"])
        trainer.fit(model, datamodule=dm)
        model = DDPMTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

model_trainer = train_model(
    T = 1000,
    lr=3e-4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_t = torch.randn((4, 3, 32, 32), device=device)
ddpm_sample = DDPMSample(model_trainer.model, T=1000)
samples = ddpm_sample(z_t)
print(samples.shape)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 8, figsize=(20, 5))
for i in range(4):
    for j in range(8):
        img = samples[i][j] * 0.5 + 0.5
        axs[i, j].imshow(img.cpu().permute(1, 2, 0).numpy())
        axs[i, j].axis("off")
plt.show()

