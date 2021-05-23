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
from collections import defaultdict

class Net(nn.Module):
    def __init__(self, in_size = (28, 28), out_size=10, hidden_size = 100, strength = 1):
        super(Net, self).__init__()
        print("MNIST Object Based Attention Model v2")
        self.out_size = out_size
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.strength = strength
        
        self.conv1 = nn.Conv2d(1, 20, 4, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 4, 1)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.h1 = self.cout(in_size[0], 4, 1)
        self.h2 = self.cout(self.h1, 2, 2)
        self.h3 = self.cout(self.h2, 4, 1)
        self.h4 = self.cout(self.h3, 2, 2)
        
        self.w1 = self.cout(in_size[1], 4, 1)
        self.w2 = self.cout(self.w1, 2, 2)
        self.w3 = self.cout(self.w2, 4, 1)
        self.w4 = self.cout(self.w3, 2, 2)
        
        self.upsample2_t = nn.Upsample(size = (self.h3, self.w3))
        self.conv2_t = nn.ConvTranspose2d(50 * 2, 20, 4, stride = 1)
        self.upsample1_t = nn.Upsample(size = (self.h1, self.w1))
        self.conv1_t = nn.ConvTranspose2d(20 * 2, 1, 4,  stride = 1)
        
        
        self.coutsize = self.h4 * self.w4 * 50
        
        self.linear_out = nn.Linear(self.coutsize, out_size)
        self.linear_out_t = nn.Linear(out_size, self.coutsize)
        
        self.dropout = nn.Dropout(0.7)
        self.latent = defaultdict(lambda: None)
        self.hidden = defaultdict(lambda: None)
        
    def cout(self, x, k, stride = 1, padding = 0): 
        return ((x - k + 2*padding) // stride + 1)

        
        
    def forward(self, x, hidden = None, out_mask = None):
        x = x * (1 - self.strength * self.hidden["conv1_in"])
        self.latent["in"] = x
        
        x = torch.relu(self.conv1(x))
        self.latent["conv1_out"] = x
        x = self.maxpool1(x)
        self.latent["maxpool1_out"] = x
        
        x = x * (1 - self.strength * self.hidden["conv2_in"])
        x = torch.relu(self.conv2(x))
        self.latent["conv2_out"] = x
        x = self.maxpool2(x)
        self.latent["maxpool2_out"] = x
        
        x = x.view(-1, self.coutsize)
        out = self.linear_out(x)
        
        self.hidden["linear_in"] = torch.tanh(self.linear_out_t(out).reshape(self.latent["maxpool2_out"].shape))
        
        self.hidden["conv2_in"] = torch.tanh(self.conv2_t(torch.cat([self.upsample2_t(self.hidden["linear_in"]), 
                                                                     self.upsample2_t(self.latent["maxpool2_out"])], 1), 
                                               output_size = self.latent["maxpool1_out"].shape))
        
        self.hidden["conv1_in"] = torch.tanh(self.conv1_t(torch.cat([self.upsample1_t(self.hidden["conv2_in"]), 
                                                                     self.upsample1_t(self.latent["maxpool1_out"])], 1), 
                                               output_size = self.latent["in"].shape))
        if out_mask is not None: 
            #zero out conv1-in by mask
            newconv1in = (1 - out_mask["conv1_in"]) * self.hidden["conv1_in"]
            #fill the masked part with 1s for full strength
            self.hidden["conv1_in"] = newconv1in + 1 * out_mask["conv1_in"]
            
            #zero out conv2-in by mask
            newconv2in = (1 - out_mask["conv2_in"]) * self.hidden["conv2_in"]
            self.hidden["conv2_in"] = newconv2in + 1 * out_mask["conv2_in"]
            
        return out
    
    def initHidden(self, device, batch_size=64): 
        self.latent = {}
        self.hidden["conv1_in"] = torch.zeros((batch_size, 1, self.in_size[0], self.in_size[1])).to(device)
        self.hidden["conv2_in"] = torch.zeros(self.maxpool1(self.conv1(self.hidden["conv1_in"])).shape).to(device)


class Runner():
    def __init__(self, net, optimizer, criterion, penalty = 1e-4, n=2, device = "cuda:3", name = "model"):
        self.device = device
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.penalty = penalty
        metrics = ["acc", "loss", "smooth_acc", "step", \
                   "val_acc", "val_step"]
        self.metrics = {x: [] for x in metrics}
        self.n = n
        self.name = name
    
    def plot(self, maskarray, rows = 2, cols = 2, x = None, save = False, train = True): 
        if train: 
            folder = "saved/plots/train"
        else:
            folder = "saved/plots/val"
        for n, masked in enumerate(maskarray): 
            plt.figure()
            fig, ax = plt.subplots(rows, cols, figsize = (cols * 5, rows*3))
            for i in range(rows * cols): 
                row = int(i / cols)
                col = i % cols
                mask = masked[i].reshape(self.net.in_size)
                ax[row, col].imshow(mask, cmap = "gray", vmin= 0, vmax = 1.1)
                ax[row, col].axis("off")
            if save: 
                savedir = "%s/%s_masked_%s"%(folder, self.name, n)
                plt.savefig("%s.png"%savedir)
            plt.show()

        plt.figure()
        if x is not None:
            fig2, ax2 = plt.subplots(rows, cols, figsize = (cols * 5, rows * 3))
            for i in range(rows * cols): 
                row = int(i / cols)
                col = i % cols
                mask = x[i].reshape(self.net.in_size)
                ax2[row, col].imshow(mask, cmap = "gray", vmin = 0, vmax = 1.1)
                ax2[row, col].axis("off")
        if save: 
            savedir = "%s/%s_orig.png"%(folder, self.name)
            plt.savefig("%s.png"%savedir)
        plt.show()
        
    
    def train(self, train_load, test_load = None, epochs = 10):
        step = 0
        for epoch in range(epochs):
            if test_load is not None: 
                self.test(test_load, step)
            for i, (x, data, labels) in enumerate(train_load):
                x = x.to(self.device)
                data = [d.to(self.device) for d in data]
                labels = [l.to(self.device) for l in labels]
                
                self.net.initHidden(self.device, x.shape[0])
                out_mask = {}
                out_mask["out"] = torch.zeros(labels[0].shape[0], 10).to(self.device)
                out_mask["conv1_in"] = torch.zeros_like(self.net.hidden["conv1_in"])
                out_mask["conv2_in"] = torch.zeros_like(self.net.hidden["conv2_in"])
                maskarray = []
                
                #set the choice mask to all false
                findselectmask = torch.zeros((x.shape[0], self.n)).type(torch.bool)
                
                #select all images
                for _ in range(self.n): 
                    #zero the optimizer
                    self.optimizer.zero_grad()
                    
                    #initialize hidden vector
                    self.net.initHidden(self.device, x.shape[0])
                    
                    #run T-1 iterations
                    for j in range(4): 
                        out = self.net(x, out_mask = out_mask)
                    
                    #our new gating mask
                    new_out_mask = {}
                    new_out_mask["conv1_in"] = ((out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5)) > 0.5).type(torch.int)
                    new_out_mask["conv2_in"] = ((out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5)) > 0.5).type(torch.int)
                    
                    #run the final iteration
                    out = self.net(x, out_mask = out_mask)

                    #get the masked input
                    masked = self.net.latent["in"]
                    maskarray.append(masked.cpu().detach().numpy().copy())


                    #we want to find which digit the network selected
                    #to do this, we calculate the MSE with each of the objects
                    #and select the index of the one with the minimum from each sample
                    #the result is a batch_size x n matrix of losses
                    findselect = [torch.sum(torch.nn.MSELoss(reduction='none')(masked, x).detach(), dim= [1, 2, 3]) for x in data]
                    findselect = torch.stack(findselect).T
                    #the findselect mask will make sure we don't choose the same object twice
                    findselect[findselectmask] = 1e10 #~infinity

                    #the selected digit is the argmin of these
                    select = torch.argmin(findselect, axis = 1)
                    findselectmask[np.arange(len(findselectmask)), select] = True

                    #concatenate y1 ... yn and index by output
                    merge_y = torch.cat([y.reshape(-1, 1) for y in labels], 1)
                    #selected out is the label corresponding with the maximum output
                    selected_out = merge_y[np.arange(len(merge_y)), select]

                    #concatenate x1 and x2, and then index by selection
                    merge_x = torch.cat(data, 1)
                    #selected x will be used for reconstruction error
                    selected_x = merge_x[np.arange(len(merge_x)), select].reshape(x.shape)

                    #reconstruction error on the masked input
                    mask_loss = nn.MSELoss()(masked, selected_x)
                    class_loss = self.criterion(out, selected_out)
                    
                    #loss is the composition of the class loss and the mask error
                    loss = class_loss + self.penalty*mask_loss
                    loss.backward()
                    self.optimizer.step()

                    _, pred = torch.max(out, 1)
                    acc = ((pred==selected_out).float().sum()/len(pred)).item()

                    self.metrics["acc"].append(acc)
                    self.metrics["smooth_acc"].append(np.mean(self.metrics["acc"][-100:]))
                    self.metrics["loss"].append(loss.item())
                    self.metrics["step"].append(step)
                    
                    #set the gating mask
                    out_mask = new_out_mask
                
                if i%100 == 0:
                    print("\t[{}/{}] \t Accuracy:{:.3}   \tLoss: {:.3f}"\
                              .format(epoch+1, epochs, self.metrics["smooth_acc"][-1],\
                                      np.mean(self.metrics["loss"][-100:])))
                    
                if step%1000 == 0: 
                    self.plot(maskarray, x = x.cpu().detach().numpy(), save = True)
                    
                step += 1
                          
    def test(self, test_load, step= None, save = True): 
        print("Validating ...")
        total_correct = 0
        total = 0
        for i, (x, data, labels) in enumerate(test_load): 
            x = x.to(self.device)
            data = [d.to(self.device) for d in data]
            labels = [l.to(self.device) for l in labels]

            self.net.initHidden(self.device, x.shape[0])
            out_mask = {}
            
            out_mask["out"] = torch.zeros(labels[0].shape[0], 10).to(self.device)
            out_mask["conv1_in"] = torch.zeros_like(self.net.hidden["conv1_in"])
            out_mask["conv2_in"] = torch.zeros_like(self.net.hidden["conv2_in"])
            maskarray=[]
            
            #set the choice mask to all false
            findselectmask = torch.zeros((x.shape[0], self.n)).type(torch.bool)
            for _ in range(self.n):
                self.net.initHidden(self.device, x.shape[0])
                
                #run T-1 iterations
                for j in range(4): 
                    out = self.net(x, out_mask = out_mask)
                
                #our new gating mask
                new_out_mask = {}
                new_out_mask["conv1_in"] = ((out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5)) > 0.5).type(torch.int)
                new_out_mask["conv2_in"] = ((out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5)) > 0.5).type(torch.int)
                
                #run our final iteration
                out = self.net(x, out_mask = out_mask)
                
                #get the masked input
                masked = self.net.latent["in"]
                maskarray.append(masked.detach().cpu().numpy().copy())

                #we want to find which digit the network selected
                #to do this, we calculate the MSE with each of the objects
                #and select the index of the one with the minimum from each sample
                #the result is a batch_size x n matrix of losses
                findselect = [torch.sum(torch.nn.MSELoss(reduction='none')(masked, x).detach(), dim= [1, 2, 3]) for x in data]
                findselect = torch.stack(findselect).T
                findselect[findselectmask] = 1e10 #~infinity
                
                #the selected digit is the argmin of these
                select = torch.argmin(findselect, axis = 1)
                findselectmask[np.arange(len(findselectmask)), select] = True

                #concatenate y1 ... yn and index by output
                merge_y = torch.cat([y.reshape(-1, 1) for y in labels], 1)
                #selected out is the label corresponding with the maximum output
                selected_out = merge_y[np.arange(len(merge_y)), select]

                _, pred = torch.max(out, 1)

                total += len(selected_out)
                total_correct += (pred==selected_out).sum().item()
                
                #set the gating mask
                out_mask = new_out_mask
        
        self.plot(maskarray, x = x.cpu().detach().numpy(), train = False, save = True)

        val_acc = total_correct/total
        if step is not None: 
            self.metrics["val_acc"].append(val_acc)
            self.metrics["val_step"].append(step)
            print("\t[Validation Acc] %.4f"%(val_acc))
        return val_acc
    
    def visualize(self, x, data, labels):
        with torch.no_grad():
            x = x.to(self.device)
            data = [d.to(self.device) for d in data]
            labels = [l.to(self.device) for l in labels]

            self.net.initHidden(self.device, x.shape[0])
            out_mask = {}
            out_mask["conv1_in"] = torch.zeros_like(self.net.hidden["conv1_in"])
            out_mask["conv2_in"] = torch.zeros_like(self.net.hidden["conv2_in"])

            masks = []
            hiddens = []
            ior = []
            
            #set the choice mask to all false
            findselectmask = torch.zeros((x.shape[0], self.n)).type(torch.bool)
            for _ in range(self.n):
                self.net.initHidden(self.device, x.shape[0])
                
                #run T-1 iterations
                for j in range(4): 
                    out = self.net(x, out_mask = out_mask)
                
                hiddens.append(self.net.hidden["conv1_in"].detach().cpu().numpy())
                
                #our new gating mask
                new_out_mask = {}
                new_out_mask["conv1_in"] = ((out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5)) > 0.5).type(torch.int)
                new_out_mask["conv2_in"] = ((out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5)) > 0.5).type(torch.int)
                
                #run our final iteration
                out = self.net(x, out_mask = out_mask)
                
                #get the masked input
                masked = self.net.latent["in"]

                #we want to find which digit the network selected
                #to do this, we calculate the MSE with each of the objects
                #and select the index of the one with the minimum from each sample
                #the result is a batch_size x n matrix of losses
                findselect = [torch.sum(torch.nn.MSELoss(reduction='none')(masked, x).detach(), dim= [1, 2, 3]) for x in data]
                findselect = torch.stack(findselect).T
                findselect[findselectmask] = 1e10 #~infinity
                
                #the selected digit is the argmin of these
                select = torch.argmin(findselect, axis = 1)
                findselectmask[np.arange(len(findselectmask)), select] = True

                #concatenate y1 ... yn and index by output
                merge_y = torch.cat([y.reshape(-1, 1) for y in labels], 1)
                #selected out is the label corresponding with the maximum output
                selected_out = merge_y[np.arange(len(merge_y)), select]
                
                #set the gating mask
                out_mask = new_out_mask

                masks.append(masked.detach().cpu().numpy())
                ior.append(out_mask["conv1_in"].detach().cpu().numpy())
            
            return masks, hiddens, ior
            
    def toshow(self, x): 
        return x[0]
        
    def get_metrics(self): 
        return self.metrics