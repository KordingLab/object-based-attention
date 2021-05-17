from data.coco_loader import *
from models.mnist_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--traindir', type= str, default = '../../../data/jordanlei/coco/images/train2017')
parser.add_argument('--anndir', type= str, default = '../../../data/jordanlei/coco/annotations/instances_train2017.json')
parser.add_argument('--metadatadir', type= str, default = 'data/metadata/metadata_2objects.p')
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

coco_dataset, metadata = get_data(root, annfile, metadatafile, size = (100, 100), strength = strength, use_supercategory = True)

train, val = get_train_val_split(coco_dataset)

