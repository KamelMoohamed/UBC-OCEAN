import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.svm import OneClassSVM


class AE(pl.LightningModule):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder network
        self.fc0 = nn.Linear(2048, 1080)
        self.fc1 = nn.Linear(1080, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

        # Decoder network
        self.fc7 = nn.Linear(2, 32)
        self.fc8 = nn.Linear(32, 64)
        self.fc9 = nn.Linear(64, 128)
        self.fc10 = nn.Linear(128, 256)
        self.fc11 = nn.Linear(256, 512)
        self.fc12 = nn.Linear(512, 1080)
        self.fc13 = nn.Linear(1080, 2048)

    def encode(self, x):
        h = torch.tanh(self.fc0(x))
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        h = torch.tanh(self.fc4(h))
        h = torch.tanh(self.fc5(h))

        return self.fc6(h)

    def decode(self, x):
        h = torch.tanh(self.fc7(x))
        h = torch.tanh(self.fc8(h))
        h = torch.tanh(self.fc9(h))
        h = torch.tanh(self.fc10(h))
        h = torch.tanh(self.fc11(h))
        h = torch.tanh(self.fc12(h))

        return self.fc13(h)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self.forward(x)
        loss = nn.MSELoss()(x_reconstructed, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self.forward(x)
        loss = nn.MSELoss()(x_reconstructed, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



class OC_svm:

    def __init__(self):
        self.svm = OneClassSVM(kernel='rbf', nu=0.1)

    def fit(self, x):
        self.svm.fit(x)

    def predict(self, x):
        return self.svm.predict(x)
