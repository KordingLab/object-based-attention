from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import inspect
import torch
from torch.utils.data import Dataset, DataLoader, random_split

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.mnist_loader import get_data as mnist_get_data
from models.mnist_model import Net as MNISTNet
from models.mnist_model import Runner as MNISTRunner

from data.coco_loader import get_data as coco_get_data
from models.coco_model import Net as COCONet
from models.coco_model import Runner as COCORunner

device =  torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def load_model_and_data(modelpath, n = 2, strength = 0.2, modeltype = 'coco', cocoroot = '', annpath = '', metadatapath = ''):
    if modeltype.lower() == "mnist":
        net = MNISTNet((28, 28+ 14 * (n-1)), strength = strength).to(device)
        net.load_state_dict(torch.load(modelpath))
        net.eval()

        _, _, test_loader = mnist_get_data(mnistdir = "../mnist", n = n, strength = strength, noise = 0.3, resample = True)
        runner = MNISTRunner(net, None, None, penalty = 1000, n=n, device = device)

        return net, runner, test_loader
    
    elif modeltype.lower() == "coco":
        net = COCONet((100, 100), strength = strength).to(device)
        net.load_state_dict(torch.load(modelpath))
        coco_dataset, metadata = coco_get_data(cocoroot, annpath, metadatapath, size = (100, 100), strength = strength, use_supercategory = True)
        test_loader = DataLoader(coco_dataset, batch_size = 32, shuffle = True, num_workers=4)

        runner = COCORunner(net, None, None, penalty = 5000, n=2, device = device)

        return net, runner, test_loader
    
def toshow(x): 
    if x.shape[0] == 1: 
        plt.figure()
        plt.imshow(x[0], cmap = "gray")
        plt.axis("off")
        plt.show()
    elif x.shape[0] == 3: 
        x = np.moveaxis(x, [0, 1, 2],[2, 0, 1])
        plt.figure()
        plt.imshow(x)
        plt.axis("off")
        plt.show()
