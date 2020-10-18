import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torchvision.datasets as datasets
from utils.cutout import Cutout


class Cifar10:
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    def __init__(self, 
                 data_folder, 
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 cutout=False,
                 cutout_length=None,
                 **kwargs):
        
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)])

        train_data = datasets.CIFAR10(root=data_folder,
                                      train=True,
                                      download=False,
                                      transform=train_transform)

        test_data = datasets.CIFAR10(root=data_folder,
                                     train=False,
                                     download=False,
                                     transform=valid_transform)
        
        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=True)
                                       
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      pin_memory=pin_memory,
                                      num_workers=num_workers,
                                      shuffle=False)
        

