from data.coco_loader import *
from models.coco_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--testpath', type= str, default = '../../../data/jordanlei/coco/images/val2017')
parser.add_argument('--annpath', type= str, default = '../../../data/jordanlei/coco/annotations/instances_val2017.json')
parser.add_argument('--metadatapath', type= str, default = 'data/metadata/cocometadata_test.p')
parser.add_argument('--strength', type= float, default = 0.3)
parser.add_argument('--out', type = str, default='attn')
parser.add_argument('--modelpath', type = str, default='saved/models/paper_coco_model.pt')

args = parser.parse_args()
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

strength = args.strength
root = args.testpath
annfile = args.annpath
metadatafile = args.metadatapath

coco_dataset, metadata = get_data(root, annfile, metadatafile, size = (100, 100), strength = strength, use_supercategory = True)
test_loader = DataLoader(coco_dataset, batch_size = 32, shuffle = True, num_workers=4)

net = Net(strength = args.strength).to(device)
net.load_state_dict(torch.load(args.modelpath))
# net.eval()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()
runner = Runner(net, optimizer, criterion, penalty = 5000, n=2, device = device)

accuracy = runner.test(test_loader)
print("Test Accuracy:", accuracy)