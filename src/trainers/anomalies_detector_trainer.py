from data_loaders.AD_dataloader import AD_dataset, AD_DataModule
from models.anomalies_detector import AE, OC_svm
from tqdm import tqdm
import torchvision.models as models
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb




def AD_train():
    #loading data for AE
    data_dir = "data/images"
    dataset = AD_dataset(data_dir)
    data_module = AD_DataModule(dataset)
    trainig_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    train_features = extract_features(trainig_loader)
    val_features = extract_features(val_loader)
    train_features = torch.tensor(np.array(train_features).reshape((np.shape(np.array(train_features))[0], -1)))
    val_features = torch.tensor(np.array(val_features).reshape((np.shape(np.array(val_features))[0], -1)))

    train_f_loader = DataLoader(train_features, batch_size=32, shuffle=True)
    val_f_loader = DataLoader(val_features, batch_size=32, shuffle=True)


    #training AE
    autoencoder = AE()

    wandb_logger = WandbLogger(name="AD_trainer", project="UBC-OCEAN")

    checkpoint_callback = ModelCheckpoint(
        dirpath="anomalies_detector/weights", save_top_k=1, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=100, logger=wandb_logger, callbacks=[checkpoint_callback]
    )

    trainer.fit(autoencoder, train_f_loader, val_f_loader)
    wandb.finish()


    #training OC_svm
    low_dim_train_features = extract_ecnoder_features(train_features, autoencoder)
    low_dim_val_features = extract_ecnoder_features(val_features, autoencoder)
    all_features = np.concatenate((low_dim_train_features, low_dim_val_features), axis=0)
    svm = OC_svm()
    svm.fit(all_features)





def extract_features(loader):
    resnet = models.resnet50(pretrained=True).to('cuda')
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    features = []
    print("Extracting features...")
    for batch in tqdm(loader):
        batch = batch.to('cuda')
        feature = resnet(batch)
        features.append(feature.cpu().detach().numpy())
    return features


def extract_ecnoder_features(features, autoencoder):
    features = autoencoder.encode(torch.tensor(features).to('cuda'))
    return features.cpu().detach().numpy()  