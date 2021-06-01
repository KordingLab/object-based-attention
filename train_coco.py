from data.coco_loader import *
from models.coco_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--trainpath', type= str, default = '../../../data/coco/images/train2017')
parser.add_argument('--annpath', type= str, default = '../../../data/coco/annotations/instances_train2017.json')
parser.add_argument('--metadatapath', type= str, default = 'data/metadata/cocometadata_train.p')
parser.add_argument('--strength', type= float, default = 0.3)
parser.add_argument('--out', type = str, default='attn')
parser.add_argument('--name', type = str, default='coco_attention_model')
parser.add_argument('--epochs', type = int, default=100)
parser.add_argument('--randomseed', type = int, default=100)
parser.add_argument('--lr', type = float, default=0.0001)
parser.add_argument('--penalty', type = float, default=5000)

args = parser.parse_args()
run_id = args.name
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")

log = run_id + "_log.txt"
f = open(log, "w")
f.write("\n")
f.close()

def printwrite(x): 
    print(x)
    f = open(log, "a")
    f.write(x + "\n")
    f.close()
    
strength = args.strength
root = args.trainpath
annfile = args.annpath
metadatafile = args.metadatapath
modelname = args.name
lr = args.lr
penalty = args.penalty

printwrite("COCO ATTENTION MODEL %s\n#OBJECTS: %s STRENGTH: %s LR: %s PHI: %s"%(modelname, 2, strength, lr, penalty))

torch.manual_seed(args.randomseed)

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

#establish weights on supercategory classes
category_count, supercategory_count = get_category_counts(train.indices, metadata["images"], metadata["annotations"])
supercat_weights = class_weight(supercategory_count)
helper_dicts = get_dicts()
sweights = {helper_dicts["supercat_to_sl"][k]: supercat_weights[k] for k in supercat_weights.keys()}
sweights = [sweights[i] for i in range(len(sweights))]

dflist = []
net = Net(strength = strength).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = lr)

class_weights = torch.Tensor(sweights).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)

runner = Runner(net, optimizer, criterion, penalty = penalty, n=2, device = device, name = modelname)
runner.train(train_loader, val_loader, epochs = args.epochs)
metric = runner.get_metrics()
metric["final_acc"] = runner.test(val_loader, save = True)

torch.save(net.state_dict(), "saved/models/%s.pt"%(modelname))
dflist.append(metric)

df = pd.DataFrame(dflist)
df.to_csv("saved/metrics/%s.csv"%(modelname))
print("[DONE]")