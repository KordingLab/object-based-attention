from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from .data.mnist_loader import get_data
from .models.mnist_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--n', type= int, default = 2)
parser.add_argument('--strength', type= float, default = 0.2)
parser.add_argument('--noise', type= float, default = 0.3)
parser.add_argument('--resample', type = bool, default = True)
parser.add_argument('--out', type = str, default='attn')
parser.add_argument('--modelpath', type = str, default='../saved/models/mnist_model.pt')


args = parser.parse_args()
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

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

def generate_bars(degrange, sigrange, xrange, yrange): 
    batches = []
    for deg in degrange:
        batch = []
        for sig in sigrange: 
            for x in xrange: 
                for y in yrange: 
                    img = bars(deg = deg, x = x, y = y, sigma = sig)
                    batch.append(np.array([img]))
        batch = torch.FloatTensor(np.array(batch))
        batches.append(batch)
    
    return batches

def calculate_activations(net, batches, im, conv_layers = ["conv_1", "conv_2"]): 
    attended_activations = defaultdict(lambda: [])
    not_attended_activations = defaultdict(lambda: [])
    noise = 0.0
    
    for batch in batches:
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
        out_mask["out"] = torch.zeros(labels[0].shape[0], 10).to(self.device)
        out_mask["conv1_in"] = torch.zeros_like(self.net.hidden["conv1_in"])
        out_mask["conv2_in"] = torch.zeros_like(self.net.hidden["conv2_in"])

        findselectmask = torch.zeros((x.shape[0], 2)).type(torch.bool)

        for _ in range(2):
            net.initHidden(device, x.shape[0])
            for _ in range(5): 
                out = net(x)
            
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
            
            out_mask = {}
            out_mask["conv1_in"] = (out_mask["conv1_in"] + (self.net.hidden["conv1_in"] < 0.5) > 0.5).type(torch.int)
            out_mask["conv2_in"] = (out_mask["conv2_in"] + (self.net.hidden["conv2_in"] < 0.5) > 0.5).type(torch.int)
    
    return attended_activations, not_attended_activations


def plot_curves(attended_activations, not_attended_activations, degrange):
    for layer in attended_activations.keys(): 
        act = np.stack(attended_activations[layer], 3)
        nonact = np.stack(not_attended_activations[layer], 3)
        
        if layer == "conv1_in":
            r = 12
            c = 8
        elif layer == "conv2_in":
            r = 4
            c = 3

        for i in range(act.shape[0]): 
            fig = plt.figure()
            plt.plot(degrange, act[i][r][c], color = (0, 1, 0, 1))
            plt.plot(degrange, nonact[i][r][c], color = (0, 0, 1, 1))
            plt.xlabel("Degrees")
            plt.ylabel("Activation")

            maxval = max(np.max(nonact[i][r][c]), np.max(act[i][r][c]))
            plt.ylim([-0.1*maxval, 1.1*maxval])

            my_folder = "tuning_curves/%s"%(layer)

            if not os.path.exists(my_folder):
                os.makedirs(my_folder)

            plt.savefig("%s/%s.png"%(my_folder, i), format = "png")

         


def __main__(): 
    print("Tuning Curves Analysis")

    print("[Step 1]\tLoading Model...")
    dataset = torchvision.datasets.MNIST("mnist", download = True,\
            transform = torchvision.transforms.ToTensor())
    
    net = Net((28, 28+14), strength = args.strength).to(device)
    net.load_state_dict(torch.load(args.modelpath))
    net.eval()
    
    print("[Step 2]\tGenerating Rotating Bars...")
    #create a "distractor digit"
    x, y = dataset[6]
    x = x.numpy().reshape(28, 28)
    im = np.zeros((28, 28+14))
    im[:, 14: x.shape[1]+14] = x

    degrange = np.linspace(0, 180, 21)
    sigrange = np.linspace(0.5, 2, 10)
    xrange = np.linspace(-15, -10, 11)
    yrange = np.linspace(-10, 10, 11)
    
    #get rotating bars
    batches = generate_bars(degrange, sigrange, xrange, yrange)

    print("[Step 3]\tRunning Network...")
    #calculate the activations
    attended, not_attended = calculate_activations(net, batches, im)

    print("[Step 4]\tPlotting Curves...")
    plot_curves(attended_activations, not_attended_activations, degrange)

    print("[Done]")

