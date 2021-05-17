from data.coco_loader import *
from models.coco_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--traindir', type= str, default = '../../../data/jordanlei/coco/images/train2017')
parser.add_argument('--anndir', type= str, default = '../../../data/jordanlei/coco/annotations/instances_train2017.json')
parser.add_argument('--metadatadir', type= str, default = 'data/metadata/cocometadata_train.p')
parser.add_argument('--strength', type= float, default = 0.9)
parser.add_argument('--out', type = str, default='attn')
parser.add_argument('--name', type = str, default='coco_model')

args = parser.parse_args()
run_id = args.name
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

strength = args.strength
root = args.traindir
annfile = args.anndir
metadatafile = args.metadatadir
modelname = args.name

coco_dataset, metadata = get_data(root, annfile, metadatafile, size = (100, 100), strength = strength, use_supercategory = True)
train, val = get_train_val_split(coco_dataset)
train_loader = DataLoader(train, batch_size = 32, shuffle = True, num_workers=4)
val_loader = DataLoader(val, batch_size = 32, shuffle = True, num_workers=4)

def get_class_weights(superclass = False): 
    helper_dicts = get_dicts()
    cat_weights = class_weight(category_count)
    supercat_weights = class_weight(supercategory_count)
    
    label_weights = defaultdict(lambda: 0)
    if superclass:
        for k in supercat_weights.keys(): 
            label = helper_dicts["supercat_to_sl"][k]
            label_weights[label] = supercat_weights[k]
    else:
        for k in cat_weights.keys():
            label = helper_dicts["cat_to_l"][k]
            label_weights[label] = cat_weights[k]
            
    return [label_weights[i] for i in range(max(label_weights.keys()) + 1)]


dflist = []
net = Net(strength = strength).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

class_weights = torch.Tensor(sweights).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)

runner = Runner(net, optimizer, criterion, penalty = 5000, n=2, device = device, name = modelname)
runner.train(train_loader, val_loader, epochs = 1)
metric = runner.get_metrics()
metric["final_acc"] = runner.test(val_loader, save = True)

torch.save(net.state_dict(), "saved/models/%s.pt"%(modelname))
dflist.append(metric)

df = pd.DataFrame(dflist)
df.to_csv("saved/metrics/%s.csv"%(modelname))
print("[DONE]")