import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split




class AD_dataset(Dataset):
    def __init__(self, img_dir):
        """
        Args:
            dataframe (DataFrame): DataFrame with annotations.
            img_dir (string): Directory with all the images.
        """
        self.img_dir = img_dir
        self.imgs_path = [os.path.join(self.img_dir, img_name) for img_name in os.listdir(self.img_dir)]
        self.transform = A.Compose(
            [
                A.Resize(256, 256),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1,
        )
        

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = self.imgs_path[idx]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image



class AD_DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1, validation_split=0.15):
        super(AD_DataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split

    def setup(self, stage=None):

        dataset_size = len(self.dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)