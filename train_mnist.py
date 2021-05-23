from data.mnist_loader import get_data
from models.mnist_model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type= int, default = 3)
parser.add_argument('--n', type= int, default = 3)
parser.add_argument('--strength', type= float, default = 0.2)
parser.add_argument('--noise', type= float, default = 0.3)
parser.add_argument('--name', type = str, default='mnist_model')
parser.add_argument('--randomseed', type = int, default=2021)
parser.add_argument('--epochs', type = int, default=10)

args = parser.parse_args()
run_id = args.name
log = run_id + "_log.txt"
device =  torch.device("cuda:%s"%(args.device) if torch.cuda.is_available() else "cpu")
f = open(log, "w")
f.write("\n")
f.close()

# torch.manual_seed(args.randomseed)

def printwrite(x): 
    print(x)
    f = open(log, "a")
    f.write(x + "\n")
    f.close()


printwrite("MNIST ATTENTION MODEL\n#DIGITS: %s STRENGTH: %s NOISE: %s"%(args.n, args.strength, args.noise))
train_loader, val_loader, _ = get_data(n = args.n, strength = args.strength, noise = args.noise, resample = True)
dflist = []

printwrite("Runing Attention Model...")

#learning rate is 1e-3
for lr in [1e-5]:

    #penalty parameter = 1e3
    for p in [0, 1e2, 1e3, 1e4, 1e5]:
        modelname = "%s_penalty_%s_lr_%s"%(run_id, p, lr)
        best_model = None
        best_score = -1

        #run for a single trial
        for k in range(5): 
            printwrite("[MODELNAME %s TRIAL %s]"%(modelname, k))
            net = Net((28, 28 + 14*(args.n -1)), strength = args.strength).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr = lr)
            criterion = nn.CrossEntropyLoss()
            runner = Runner(net, optimizer, criterion, penalty = p, n = args.n, device = device, name = modelname)
            runner.train(train_loader, val_loader, epochs = args.epochs)

            metric = runner.get_metrics()
            metric["final_acc"] = runner.test(val_loader, save = True)
            metric["penalty"] = p
            metric["lr"] = lr
            metric["name"] = modelname

            if metric["final_acc"] > best_score: 
                best_score = metric["final_acc"]
                torch.save(net.state_dict(), "saved/models/%s.pt"%(modelname))
                printwrite("\t[SAVE] New Best Model For %s With Score %s -- Saved"%(p, best_score))

            dflist.append(metric)

df = pd.DataFrame(dflist)
df.to_csv("saved/metrics/%s.csv"%(run_id))
printwrite("[DONE]")