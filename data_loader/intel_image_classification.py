import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from utils.cutout import Cutout

class IntelImageClassification:
    IIC_MEAN = [.5] * 3
    IIC_STD = [.5] * 3
    def __init__(self, 
                 train_folder,
                 test_folder, 
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 cutout=False,
                 cutout_length=None,
                 **kwargs):
        
        train_transform = transforms.Compose([transforms.RandomCrop(150, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.IIC_MEAN, self.IIC_STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.IIC_MEAN, self.IIC_STD)])

        train_data = datasets.ImageFolder(root=train_folder,
                                          transform=train_transform)

        test_data = datasets.ImageFolder(root=test_folder,
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

        # h, w = 0, 0
        # for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        #     if batch_idx == 0:
        #         h, w = inputs.size(2), inputs.size(3)
        #         print(inputs.min(), inputs.max())
        #         chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        #     else:
        #         chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
        # mean = chsum/len(train_data)/h/w
        # print('mean: %s' % mean.view(-1))

        # chsum = None
        # for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        #     if batch_idx == 0:
        #         chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        #     else:
        #         chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        # std = torch.sqrt(chsum/(len(train_data) * h * w - 1))
        # print('std: %s' % std.view(-1))

        # print('Done!')