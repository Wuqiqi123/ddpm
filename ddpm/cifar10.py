import lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/"):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

    def prepare_data(self):
        """
        This method is called only once and on only one GPU. It's used to perform any data download or preparation steps.
        """
        print("Preparing data...")

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset = CIFAR10(root=self.data_dir, train=True, transform=self.train_transform, download=True)
            val_dataset = CIFAR10(root=self.data_dir, train=True, transform=self.train_transform, download=True)
            self.train_set, _ = random_split(train_dataset, [45000, 5000])
            _, self.val_set = random_split(val_dataset, [45000, 5000])

        if stage == "test":
            self.test_set = CIFAR10(root=self.data_dir, train=False, transform=self.train_transform, download=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=32)