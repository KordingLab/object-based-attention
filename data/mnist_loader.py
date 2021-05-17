import torchvision
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd

class MultiMNIST(Dataset):
    def __init__(self, mnist_data, offset = 14, train = True, n = 3, strength = 0.5, noise = 0.0, resample = False): 
        super(MultiMNIST, self).__init__()
        self.train = train
        self.dataset = mnist_data
        self.offset = offset
        self.n = n
        self.strength = strength
        self.noise = noise
        self.resample = resample
        
    def __getitem__(self, idx):
        strength = self.strength
        offset = self.offset
        n = self.n
        
        xlist = []
        ylist = []
        
        data = []
        labels = []
        
        x, y = self.dataset.__getitem__(idx)
        xlist.append(x)
        ylist.append(y)
        
        for i in range(n-1):
            x, y = self.dataset.__getitem__(np.random.randint(len(self.dataset)))
            if not self.resample: 
                while y in ylist: 
                    x, y = self.dataset.__getitem__(np.random.randint(len(self.dataset)))
            xlist.append(x)
            ylist.append(y)
        
        
        order = np.arange(n)
        np.random.shuffle(order)
        
        shape = (x.shape[0], x.shape[1], x.shape[2] + (n-1) * offset)
        
        
        off = 0
        for index in order: 
            x = xlist[index]
            y = ylist[index]
            
            d = torch.zeros(shape)
            d[:, :, off : off + x.shape[2]] = x
            data.append(torch.FloatTensor(d))
            labels.append(y)
            
            off = off + offset
        
        x = torch.sum(torch.stack(data), dim = 0)
        x = torch.clamp(x, 0, 1)
        
        noise_matrix = torch.FloatTensor(np.random.random(x.shape) * self.noise)
        n_mask = x <= 0
        noise_matrix = noise_matrix * n_mask
        x = x + noise_matrix
           
        data = [(1-strength) * x +  strength * d for d in data]

        return x, data, labels
        
    def __len__(self): 
        return self.dataset.__len__()

def get_data(verbose = True, n = 2, strength = 1, noise = 0.0, resample = False): 
    dataset = torchvision.datasets.MNIST("mnist", download = True,\
          transform = torchvision.transforms.ToTensor())
    train_size = int(len(dataset) * 0.7)
    train, val = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_mmnist = MultiMNIST(train, train = True, n = n, strength = strength, noise = noise, resample = resample)
    val_mmnist = MultiMNIST(val, train = False, n = n, strength = strength, noise = noise, resample = resample)
    
    test = torchvision.datasets.MNIST("mnist", train = False, download = True,\
          transform = torchvision.transforms.ToTensor())
    test_mmnist = MultiMNIST(test, train = False, n = n, strength = strength, noise = noise, resample = resample)

    if verbose: 
        print("Data Loaded:")
        print("\tTraining Samples: ", len(train_mmnist))
        print("\tValidation Samples: ", len(val_mmnist))
        print("\tTesting Samples: ",len(test_mmnist))
        print("\tObjects: ", n, "\tStrength: ", strength, "\tNoise: ", noise, "\tResample: ", resample)

    train_loader = DataLoader(train_mmnist, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val_mmnist, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_mmnist, batch_size = 64, shuffle = True)
    return train_loader, val_loader, test_loader