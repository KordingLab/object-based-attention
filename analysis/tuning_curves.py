from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import inspect
from scipy import stats
import seaborn as sns

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.mnist_loader import get_data
from models.mnist_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 2)
parser.add_argument('--n', type= int, default = 2)
parser.add_argument('--strength', type= float, default = 0.2)
parser.add_argument('--noise', type= float, default = 0.3)
parser.add_argument('--resample', type = bool, default = True)
parser.add_argument('--modelpath', type = str, default='../saved/models/paper_mnist_model.pt')


args = parser.parse_args()
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")
strength = args.strength

def toshow(x): 
    plt.figure()
    plt.imshow(x.detach().cpu().numpy(), cmap ='gray')
    plt.axis("off")

def bars(deg = 0, sigma = 1.5, x = 0, y = 0, shape = (28, 28+14), length = 20):
    normal = np.linspace(-10, 10, 28)
    line = stats.norm.pdf(normal, 0, sigma)
    img = np.array([line]*length).transpose()
    img = img/(np.max(img)) * 1
    img = Image.fromarray(img)
    img = img.rotate(deg)
    img = img.transform((shape[1], shape[0]), Image.AFFINE, \
                        (1, 0, int(-x + length/2 - shape[1]/2), 0, 1, y))
    img = np.array(img)
    return img

def generate_bars(degrange, sigrange, xrange, yrange, shape = (28, 28+14)): 
    batches = []
    for deg in degrange:
        batch = []
        for sig in sigrange: 
            for x in xrange: 
                for y in yrange: 
                    img = bars(deg = deg, x = x, y = y, sigma = sig, shape = shape)
                    batch.append(np.array([img]))
        batch = torch.FloatTensor(np.array(batch))
        batches.append(batch)
    
    return batches

def calculate_activations(net, batches, im, conv_layers = ["conv1_out", "conv2_out"]):
    with torch.no_grad():
        attended_activations = defaultdict(lambda: [])
        not_attended_activations = defaultdict(lambda: [])

        noise = 0.0

        for batch in batches:
            attended = defaultdict(lambda: [])
            not_attended = defaultdict(lambda: [])
            
            digit = torch.stack([im]*batch.shape[0])
            x = batch + digit
            noise_matrix = torch.FloatTensor(np.random.random(x.shape) * noise)
            n_mask = x <= 0.1
            noise_matrix = noise_matrix * n_mask
            x = x + noise_matrix

            bar = x * (1-strength) + strength * batch
            digit = x * (1-strength) + strength * digit

            x = x.to(device)
            bar = bar.to(device)
            digit = digit.to(device)

            net.initHidden(device, x.shape[0])
            out_mask = {}
            out_mask["conv1_in"] = torch.zeros_like(net.hidden["conv1_in"])
            out_mask["conv2_in"] = torch.zeros_like(net.hidden["conv2_in"])

            findselectmask = torch.zeros((x.shape[0], 2)).type(torch.bool)

            for _ in range(2):
                net.initHidden(device, x.shape[0])
                
                #run T-1 iterations
                for _ in range(4): 
                    out = net(x, out_mask = out_mask)
                
                #our new gating mask
                new_out_mask = {}
                new_out_mask["conv1_in"] = ((out_mask["conv1_in"] + (net.hidden["conv1_in"] < 0.5)) > 0.5).type(torch.int)
                new_out_mask["conv2_in"] = ((out_mask["conv2_in"] + (net.hidden["conv2_in"] < 0.5)) > 0.5).type(torch.int)
                
                #run our final iteration
                out = net(x, out_mask = out_mask)

                data = [bar, digit]
                
                masked = net.latent["in"]
                findselect = [torch.sum(torch.nn.MSELoss(reduction='none')(masked, x).detach(), dim= [1, 2, 3]) for x in data]
                findselect = torch.stack(findselect).T
                findselect[findselectmask] = 1e10

                select = torch.argmin(findselect, axis = 1).detach().cpu()
                findselectmask[np.arange(len(findselectmask)), select] = True

                attended_index = select == 0
                not_attended_index = select == 1

                for layer in conv_layers: 
                    conv_out = net.latent[layer].detach()
                    attended[layer].append(conv_out[attended_index])
                    not_attended[layer].append(conv_out[not_attended_index])

                #set the new gating mask
                out_mask = new_out_mask

            for layer in conv_layers: 
                att = torch.cat(attended[layer], 0)
                not_att = torch.cat(not_attended[layer], 0)

                max_att = torch.max(att, 0).values.detach().cpu().numpy()
                max_not_att = torch.max(not_att, 0).values.detach().cpu().numpy()

                attended_activations[layer].append(max_att)
                not_attended_activations[layer].append(max_not_att)


        return attended_activations, not_attended_activations


def plot_curves(attended_activations, not_attended_activations, degrange):
    sns.set_context("poster", font_scale=2.3)
    for layer in attended_activations.keys(): 
        act = np.stack(attended_activations[layer], 3)
        nonact = np.stack(not_attended_activations[layer], 3)
        
        if layer == "conv1_out":
            r = 12
            c = 8
        elif layer == "conv2_out":
            r = 4
            c = 3

        for i in range(act.shape[0]): 
            plt.plot(degrange, act[i][r][c], color = (14/255, 77/255, 179/255, 1))
            plt.plot(degrange, nonact[i][r][c], color = (212/255, 78/255, 78/255, 1))
            plt.xlabel("Degrees")
            plt.ylabel("Activation")

            maxval = max(np.max(nonact[i][r][c]), np.max(act[i][r][c]))
            plt.ylim([-0.1*maxval, 1.1*maxval])

            my_folder = "tuning_curves/%s"%(layer)

            if not os.path.exists(my_folder):
                os.makedirs(my_folder)

            plt.savefig("%s/%s.svg"%(my_folder, i), format = "svg")

         

print("Tuning Curves Analysis")

print("[Step 1]\tLoading Model...")
dataset = torchvision.datasets.MNIST("../mnist", download = True,\
        transform = torchvision.transforms.ToTensor())

print(args.modelpath)
shape = (28, 28 + 14*(args.n-1))
net = Net(shape, strength = args.strength).to(device)
net.load_state_dict(torch.load(args.modelpath))
net.eval()

print("[Step 2]\tGenerating Rotating Bars...")
#create a "distractor digit"
x, y = dataset[6]
x = x.numpy().reshape(28, 28)
im = np.zeros(shape)
im[:, 14: x.shape[1]+14] = x
im = torch.FloatTensor(im.reshape(1, shape[0], shape[1]))

degrange = np.linspace(0, 180, 21)
sigrange = np.linspace(0.5, 2, 6)
xrange = np.linspace(-15, -10, 11)
yrange = np.linspace(-10, 10, 11)

#get rotating bars
batches = generate_bars(degrange, sigrange, xrange, yrange, shape = shape)

print("[Step 3]\tRunning Network...")
#calculate the activations
attended_activations, not_attended_activations = calculate_activations(net, batches, im)

print("[Step 4]\tPlotting Curves...")
plot_curves(attended_activations, not_attended_activations, degrange)

print("[Done]")

