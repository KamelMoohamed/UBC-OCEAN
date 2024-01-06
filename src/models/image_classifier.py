import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import models


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=5):
        super(ImageClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
        )

        self.train_accuracy = Accuracy(task="MULTICLASS", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="MULTICLASS", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.train_accuracy(y_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        val_acc = self.val_accuracy(y_hat, y)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer
