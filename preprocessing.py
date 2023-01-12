import torchvision
from torch.utils.data import random_split, DataLoader

ROOT_FOLDER = "archive/raw-img"


def load_imagefolder(root_folder=ROOT_FOLDER, size=256):
    
    dataset = torchvision.datasets.ImageFolder(root_folder,
                                               transform=torchvision.transforms.Compose(
                                                   [torchvision.transforms.Resize((size, size)),
                                                    torchvision.transforms.ToTensor()]
                                               ))
    return dataset

def train_test_split(dataset, train_size=0.8):
    train_set, test_set = random_split(dataset, lengths=(train_size, 1 - train_size))
    datasets = {
        "train": train_set,
        "test": test_set
    }
    return datasets
