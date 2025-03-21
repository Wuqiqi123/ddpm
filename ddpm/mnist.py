import lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/"):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]
        )

    def prepare_data(self):
        """
        This method is called only once and on only one GPU. It's used to perform any data download or preparation steps.
        """
        print("Preparing data...")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MNIST(root=self.data_dir, train=True, transform=self.train_transform, download=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle = True)