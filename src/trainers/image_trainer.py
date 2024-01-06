import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_loaders.images_dataset import ImagesDataset
from src.models.image_classifier import ImageClassifier


def train():
    batch_size = 16

    X = np.load("data/images.npy")
    y = np.load("data/labels.npy")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_ds = ImagesDataset(X_train, y_train)
    val_ds = ImagesDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    wandb_logger = WandbLogger(name="ImageTrainer", project="UBC-OCEAN")

    model = ImageClassifier()

    checkpoint_callback = ModelCheckpoint(
        dirpath="classification/weights", save_top_k=1, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=30, logger=wandb_logger, callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
