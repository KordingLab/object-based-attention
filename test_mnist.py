from data.mnist_loader import get_data
from models.mnist_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--n', type= int, default = 2)
parser.add_argument('--strength', type= float, default = 0.2)
parser.add_argument('--noise', type= float, default = 0.3)
parser.add_argument('--modelpath', type = str, default='saved/models/paper_mnist_model.pt')

args = parser.parse_args()
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_data(n = args.n, strength = args.strength, noise = args.noise, resample = True)
net = Net((28, 28+14), strength = args.strength).to(device)
net.load_state_dict(torch.load(args.modelpath))
net.eval()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
runner = Runner(net, optimizer, criterion, penalty = 1000, n=args.n, device = device)

accuracy = runner.test(test_loader)
print("Test Accuracy:", accuracy)