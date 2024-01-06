from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImagesDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transforms(image)

        label = self.labels[idx]

        return image, label
