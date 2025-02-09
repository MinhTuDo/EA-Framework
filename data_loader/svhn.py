import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as datasets
from utils.cutout import Cutout, load
import torch

from scipy.io import loadmat


class SVHN:
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]
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
                                              transforms.Normalize(self.SVHN_MEAN, self.SVHN_STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.SVHN_MEAN, self.SVHN_STD)])

        train_data = datasets.SVHN(root=data_folder,
                                      split='train',
                                      download=True,
                                      transform=train_transform)

        extra_data = datasets.SVHN(root=data_folder,
                                    split='extra',
                                    download=True,
                                    transform=train_transform)
        
        train_data = ConcatDataset([train_data, extra_data])

        test_data = datasets.SVHN(root=data_folder,
                                     split='test',
                                     download=True,
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