import torchvision
import torch
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, f1_score


class Net(nn.Module):
    def __init__(self, in_size = (100, 100), out_size=12, hidden_size = 100, strength = 1):
        super(Net, self).__init__()
        print("COCO Object Based Attention Model")
        self.out_size = out_size
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.strength = strength
        
        channels = [3, 30, 60, 100, 20]
        self.channels = channels
        
        self.conv1 = nn.Conv2d(channels[0], channels[1], 4, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(channels[1], channels[2], 4, 1)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(channels[2], channels[3], 4, 1)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(channels[3], channels[4], 4, 1)
        self.maxpool4 = nn.MaxPool2d(2)
        
        self.h1_1 = self.cout(in_size[0], 4, 1)
        self.h1_2 = self.cout(self.h1_1, 2, 2)
        self.h2_1 = self.cout(self.h1_2, 4, 1)
        self.h2_2 = self.cout(self.h2_1, 2, 2)
        self.h3_1 = self.cout(self.h2_2, 4, 1)
        self.h3_2 = self.cout(self.h3_1, 2, 2)
        self.h4_1 = self.cout(self.h3_2, 4, 1)
        self.h4_2 = self.cout(self.h4_1, 2, 2)
        
        self.w1_1 = self.cout(in_size[1], 4, 1)
        self.w1_2 = self.cout(self.w1_1, 2, 2)
        self.w2_1 = self.cout(self.w1_2, 4, 1)
        self.w2_2 = self.cout(self.w2_1, 2, 2)
        self.w3_1 = self.cout(self.w2_2, 4, 1)
        self.w3_2 = self.cout(self.w3_1, 2, 2)
        self.w4_1 = self.cout(self.w3_2, 4, 1)
        self.w4_2 = self.cout(self.h4_1, 2, 2)
        
        
        self.upsample4_t = nn.Upsample(size = (self.h4_1, self.w4_1), mode = 'bicubic', align_corners=False)
        self.conv4_t = nn.ConvTranspose2d(channels[4] * 2, channels[3], 4, stride = 1)
        self.upsample3_t = nn.Upsample(size = (self.h3_1, self.w3_1), mode = 'bicubic', align_corners=False)
        self.conv3_t = nn.ConvTranspose2d(channels[3] * 2, channels[2], 4, stride = 1)
        self.upsample2_t = nn.Upsample(size = (self.h2_1, self.w2_1), mode = 'bicubic', align_corners=False)
        self.conv2_t = nn.ConvTranspose2d(channels[2] * 2, channels[1], 4, stride = 1)
        self.upsample1_t = nn.Upsample(size = (self.h1_1, self.w1_1), mode = 'bicubic', align_corners=False)
        self.conv1_t = nn.ConvTranspose2d(channels[1] * 2, channels[0], 4,  stride = 1)
        
        
        self.coutsize = self.h4_2 * self.w4_2 * channels[4]
        
        self.linear_out = nn.Linear(self.coutsize, out_size)
        self.linear_out_t = nn.Linear(out_size, self.coutsize)
        
        self.dropout = nn.Dropout(0.7)
        self.latent = defaultdict(lambda: None)
        self.hidden = defaultdict(lambda: None)
        self.leakyrelu = nn.LeakyReLU(0.05)
        
    def cout(self, x, k, stride = 1, padding = 0): 
        return ((x - k + 2*padding) // stride + 1)

    def forward(self, x, hidden = None, out_mask = None):
        shape = x.shape
        masked = x
        
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
        
        x = x * (1 - self.strength * self.hidden["conv3_in"])
        x = torch.relu(self.conv3(x))
        self.latent["conv3_out"] = x
        x = self.maxpool3(x)
        self.latent["maxpool3_out"] = x
        
        x = x * (1 - self.strength * self.hidden["conv4_in"])
        x = torch.relu(self.conv4(x))
        self.latent["conv4_out"] = x
        x = self.maxpool4(x)
        self.latent["maxpool4_out"] = x
        
        x = x.view(-1, self.coutsize)
        out = self.linear_out(x)
        
        self.hidden["linear_in"] = torch.tanh(self.linear_out_t(out).reshape(self.latent["maxpool4_out"].shape))
        
        self.hidden["conv4_in"] = torch.tanh(self.conv4_t(torch.cat([self.upsample4_t(self.hidden["linear_in"]), 
                                                                     self.upsample4_t(self.latent["maxpool4_out"])], 1), 
                                               output_size = self.latent["maxpool3_out"].shape))
        
        self.hidden["conv3_in"] = torch.tanh(self.conv3_t(torch.cat([self.upsample3_t(self.hidden["conv4_in"]), 
                                                                     self.upsample3_t(self.latent["maxpool3_out"])], 1), 
                                               output_size = self.latent["maxpool2_out"].shape))
        
        self.hidden["conv2_in"] = torch.tanh(self.conv2_t(torch.cat([self.upsample2_t(self.hidden["conv3_in"]), 
                                                                     self.upsample2_t(self.latent["maxpool2_out"])], 1), 
                                               output_size = self.latent["maxpool1_out"].shape))
        
        self.hidden["conv1_in"] = torch.tanh(self.conv1_t(torch.cat([self.upsample1_t(self.hidden["conv2_in"]), 
                                                                     self.upsample1_t(self.latent["maxpool1_out"])], 1), 
                                               output_size = self.latent["in"].shape))
        if out_mask is not None: 
            #color invariant masking
#             self.hidden["conv1_in"] = torch.stack([torch.min(self.hidden["conv1_in"], 1).values] * self.hidden["conv1_in"].shape[1], 1)

            #zero out conv1-in by mask
            newconv1in = (1 - out_mask["conv1_in"]) * self.hidden["conv1_in"]
            #fill the masked part with 1s for full strength
            self.hidden["conv1_in"] = newconv1in + 1 * out_mask["conv1_in"]
            
            #zero out conv2-in by mask
            newconv2in = (1 - out_mask["conv2_in"]) * self.hidden["conv2_in"]
            self.hidden["conv2_in"] = newconv2in + 1 * out_mask["conv2_in"]
            
            #zero out conv2-in by mask
            newconv3in = (1 - out_mask["conv3_in"]) * self.hidden["conv3_in"]
            self.hidden["conv3_in"] = newconv3in + 1 * out_mask["conv3_in"]
            
            #zero out conv2-in by mask
            newconv4in = (1 - out_mask["conv4_in"]) * self.hidden["conv4_in"]
            self.hidden["conv4_in"] = newconv4in + 1 * out_mask["conv4_in"]
            
        return out
    
    def initHidden(self, device, batch_size=64): 
        self.latent = {}
        self.hidden["conv1_in"] = torch.zeros((batch_size, 3, self.in_size[0], self.in_size[1])).to(device)
        self.hidden["conv1_in_dsp"] = torch.zeros((batch_size, 3, self.in_size[0], self.in_size[1])).to(device)
        self.hidden["conv2_in"] = torch.zeros(batch_size, self.channels[1], self.h1_2, self.w1_2).to(device)
        self.hidden["conv3_in"] = torch.zeros(batch_size, self.channels[2], self.h2_2, self.w2_2).to(device)
        self.hidden["conv4_in"] = torch.zeros(batch_size, self.channels[3], self.h3_2, self.w3_2).to(device)

        
class Runner():
    def __init__(self, net, optimizer, criterion, penalty = 1e-4, n=3, device = "cuda:3", name = "model"):
        self.device = device
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.penalty = penalty
        self.metrics = defaultdict(lambda: [])
        self.n = n
        self.name = name
    
    def plot(self, maskarray, rows = 2, cols = 2, x = None, save = False, train = True): 
        if train: 
            folder = "saved/plots"
        else:
            folder = "saved/plots"
        for n, masked in enumerate(maskarray): 
            plt.figure()
            fig, ax = plt.subplots(rows, cols, figsize = (cols * 5, rows*3))
            for i in range(rows * cols): 
                row = int(i / cols)
                col = i % cols
                mask = np.moveaxis(masked[i], [0, 1, 2],[2, 0, 1])
                ax[row, col].imshow(mask)
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
                mask = np.moveaxis(x[i], [0, 1, 2],[2, 0, 1])
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
                out_mask["conv3_in"] = torch.zeros_like(self.net.hidden["conv3_in"])
                out_mask["conv4_in"] = torch.zeros_like(self.net.hidden["conv4_in"])
                maskarray = []
                
                #set the choice mask to all false
                findselectmask = torch.zeros((x.shape[0], self.n)).type(torch.bool)
                loss = 0
                
                #select all images
                for _ in range(self.n): 
                    #zero the optimizer
                    self.optimizer.zero_grad()
                    
                    #initialize hidden vector
                    self.net.initHidden(self.device, x.shape[0])
                    for j in range(5): 
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
                    merge_x = torch.stack(data, 1)
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
                    
                    categories = np.arange(self.net.out_size)
                    predictions = pred.detach().cpu().tolist()
                    ground_truth = selected_out.detach().cpu().tolist()
                    f1 = f1_score(ground_truth, predictions, average = 'macro')

                    self.metrics["acc"].append(acc)
                    self.metrics["smooth_acc"].append(np.mean(self.metrics["acc"][-100:]))
                    self.metrics["f1"].append(f1)
                    self.metrics["smooth_f1"].append(np.mean(self.metrics["f1"][-100:]))
                    self.metrics["loss"].append(loss.item())
                    self.metrics["step"].append(step)
                    
                    out_mask["conv1_in"] = (out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5) > 0.5).type(torch.int)
                    out_mask["conv2_in"] = (out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5) > 0.5).type(torch.int)
                    out_mask["conv3_in"] = (out_mask["conv3_in"] + (self.net.hidden["conv3_in"] < 0.5) > 0.5).type(torch.int)
                    out_mask["conv4_in"] = (out_mask["conv4_in"] + (self.net.hidden["conv4_in"] < 0.5) > 0.5).type(torch.int)
                    
                
                
                
                if i%10 == 0:
                    print("\t[{}/{}] \t Accuracy:{:.3f}\tF1:{:.3f}\tLoss: {:.3f}"\
                              .format(epoch+1, epochs, self.metrics["smooth_acc"][-1],\
                                      self.metrics["smooth_f1"][-1],\
                                      np.mean(self.metrics["loss"][-100:])))
                    
                if step%100 == 0: 
                    self.plot(maskarray, x = x.cpu().detach().numpy(), save = True)
                    
                step += 1
                          
    def test(self, test_load, step= None, save = True): 
        print("Validating ...")
        total_correct = 0
        ground_truth = []
        predictions = []
        
        for i, (x, data, labels) in enumerate(test_load): 
            x = x.to(self.device)
            data = [d.to(self.device) for d in data]
            labels = [l.to(self.device) for l in labels]

            self.net.initHidden(self.device, x.shape[0])
            out_mask = {}
            
            out_mask["out"] = torch.zeros(labels[0].shape[0], 10).to(self.device)
            out_mask["conv1_in"] = torch.zeros_like(self.net.hidden["conv1_in"])
            out_mask["conv2_in"] = torch.zeros_like(self.net.hidden["conv2_in"])
            out_mask["conv3_in"] = torch.zeros_like(self.net.hidden["conv3_in"])
            out_mask["conv4_in"] = torch.zeros_like(self.net.hidden["conv4_in"])
            maskarray=[]
            
            #set the choice mask to all false
            findselectmask = torch.zeros((x.shape[0], self.n)).type(torch.bool)
            for _ in range(self.n):
                self.net.initHidden(self.device, x.shape[0])
                for j in range(5): 
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

                ground_truth += selected_out.detach().cpu().tolist()
                predictions += pred.detach().cpu().tolist()
                
                out_mask["conv1_in"] = (out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5) > 0.5).type(torch.int)
                out_mask["conv2_in"] = (out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5) > 0.5).type(torch.int)
                out_mask["conv3_in"] = (out_mask["conv3_in"] + (self.net.hidden["conv3_in"] < 0.5) > 0.5).type(torch.int)
                out_mask["conv4_in"] = (out_mask["conv4_in"] + (self.net.hidden["conv4_in"] < 0.5) > 0.5).type(torch.int)
                
        self.plot(maskarray, x = x.cpu().detach().numpy(), train = False, save = True)

        val_acc = (np.array(ground_truth) == np.array(predictions)).mean()
        categories = np.arange(self.net.out_size)
        cmatrix = confusion_matrix(ground_truth, predictions, labels = categories, normalize = 'true')
        f1 = f1_score(ground_truth, predictions, labels = categories, average = 'macro')
        plt.imshow(cmatrix)
        plt.show()
        
        if step is not None: 
            self.metrics["val_acc"].append(val_acc)
            self.metrics["val_step"].append(step)
            self.metrics["val_f1"].append(f1)
            print("\t[Validation] Acc %.4f\tF1 %.4f"%(val_acc, f1))
        return val_acc
    
    def get_metrics(self): 
        return self.metrics