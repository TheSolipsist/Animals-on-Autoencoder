import torch
import torchvision
from torch.utils.data import random_split, Dataset
from time import perf_counter

ROOT_FOLDER = "archive/raw-img"

class LoadedDataset(Dataset):
    """
    Image dataset class for loading an ImageFolder dataset to memory
    """
    def __init__(self, folder: torchvision.datasets.DatasetFolder, device: torch.device = "cpu"):
        self.device = device
        self.n = len(folder)
        t_start = perf_counter()
        self.data, self.labels = self.__load_data(folder)
        t_end = perf_counter()
        print(f"Loaded data in {(t_end - t_start):.2f} seconds")
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    def __load_data(self, folder: torchvision.datasets.DatasetFolder) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the data from the DatasetFolder and return the data, labels tuple

        Args:
            folder (DatasetFolder): The DatasetFolder to extract  data from
        """
        first_sample = folder[0][0] # Used to extract shape and dtype
        # torch.Size works as a tuple, so (1,) + (2, 3) makes (1, 2, 3)
        data = torch.empty(size=(self.n,) + first_sample.shape, device=self.device, dtype=first_sample.dtype, requires_grad=False)
        labels = torch.empty(size=(self.n,), device=self.device, dtype=torch.int, requires_grad=False)
        for i in range(self.n):
            loaded_sample = folder[i]
            data[i] = loaded_sample[0]
            labels[i] = loaded_sample[1]
        return data, labels
    
    def change_device(self, device: torch.device):
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
    
def load_images(root_folder=ROOT_FOLDER, size=256, device="cpu"):
    return LoadedDataset(get_imagefolder(ROOT_FOLDER, size), device=device)

def train_test_split(dataset, train_size=0.8):
    train_set, test_set = random_split(dataset, lengths=(train_size, 1 - train_size))
    datasets = {
        "train": train_set,
        "test": test_set
    }
    return datasets

def get_imagefolder(root_folder=ROOT_FOLDER, size=256):
    folder = torchvision.datasets.ImageFolder(root_folder,
                                               transform=torchvision.transforms.Compose(
                                                   [torchvision.transforms.Resize((size, size)),
                                                    torchvision.transforms.ToTensor()]
                                             ))
    return folder