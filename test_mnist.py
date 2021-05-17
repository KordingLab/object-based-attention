from data.mnist_loader import get_data
from models.mnist_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--n', type= int, default = 2)
parser.add_argument('--strength', type= float, default = 0.2)
parser.add_argument('--noise', type= float, default = 0.3)
parser.add_argument('--resample', type = bool, default = True)
parser.add_argument('--out', type = str, default='attn')
parser.add_argument('--name', type = str, default='mnist_model')

args = parser.parse_args()
run_id = args.name
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_data(n = args.n, strength = args.strength, noise = args.noise, resample = True)
net = Net((28, 28+14), strength = strength).to(device)
net.load_state_dict(torch.load("models/model_strength_0.2_penalty_1000.0_lr_0.001.pt"))
net.eval()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
runner = Runner(net, optimizer, criterion, penalty = 1000, n=args.n, device = device)

accuracy = runner.test(test_loader)
print("Test Accuracy:", accuracy)