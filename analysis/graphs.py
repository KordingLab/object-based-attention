from helper import *
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_final(x): 
    '''
    Gets the last element of a list from string form
    '''
    train_accs = [float(x) for x in x[1: -1].split(",")] 
    return train_accs[-1]

def convert(x): 
    '''
    Converts a stringified list into a list of floats
    '''
    return np.array([float(x) for x in x[1: -1].split(",")])

def files_to_df(metric_files): 
    dfs = []
    for metric in metric_files:
        file = "../saved/metrics/" + metric
        df = pd.read_csv(file)
        dfs.append(df.copy())
        
    df = pd.concat(dfs)
    df["train_acc"] = df["smooth_acc"].apply(lambda x: get_final(x))
    df["convert_train"] = df["smooth_acc"].apply(lambda x: convert(x))
    df["convert_train_steps"] = df["step"].apply(lambda x: convert(x))

    if "val_f1" in df.columns: 
        df["train_f1"] = df["smooth_f1"].apply(lambda x: get_final(x))
        df["final_f1"] = df["val_f1"].apply(lambda x: get_final(x))
        
    df["convert_val"] = df["val_acc"].apply(lambda x: convert(x))
    df["convert_val_steps"] = df["val_step"].apply(lambda x: convert(x))
    
    return df

def format_lineplot(metric, steps = None, every = 1):
    dflist = []
    for m in metric:
        if steps is None: 
            steps = np.arange(len(m))
        dflist.append(pd.DataFrame({"step": np.array(steps)[::every], "metric": np.array(m)[::every]}))
    df_comp = pd.concat(dflist)
    return df_comp

def plot_train_val_curves(df, save = False): 
    sns.set_context("poster")
    plt.figure(figsize=(20, 10))
    for penalty in set(df["penalty"]):
        vals = np.array(df[df["penalty"]==penalty]["convert_val"])
        val_steps = np.array(df[df["penalty"]==penalty]["convert_val_steps"])
        val_df = format_lineplot(vals, val_steps[0])
        
        if penalty != 0:
            p = "10e%s"%np.log10(penalty)
        else: 
            p = 0
        sns.lineplot(data = val_df, x = "step", y = "metric", label = "Penalty %s"%p)
    plt.xlabel("Steps")
    plt.ylabel("Validation Accuracy")
    
    if save: 
        plt.savefig("graphs/train_curves.svg")

        

    sns.set_context("poster")
    plt.figure(figsize=(20, 10))
    for penalty in set(df["penalty"]):
        trains = np.array(df[df["penalty"]==penalty]["convert_train"])
        train_steps = np.array(df[df["penalty"]==penalty]["convert_train_steps"])
        train_df = format_lineplot(trains, train_steps[0], every = 100)
        
        if penalty != 0:
            p = "10e%s"%np.log10(penalty)
        else: 
            p = 0
        sns.lineplot(data = train_df, x = "step", y = "metric", label = "Penalty %s"%p)
    
    plt.xlabel("Steps")
    plt.ylabel("Training Accuracy")

    if save: 
        plt.savefig("graphs/val_curves.svg")

def plot_boxplots(df, save = False): 
    plt.figure(figsize = (10, 5))
    sns.set_context("poster")
    sns.barplot(data = df, x = "penalty", y = "train_acc", capsize=.2)
    plt.xlabel("Loss Parameter Φ")
    plt.ylabel("Training Accuracy")
    plt.ylim([0.8, 1.0])
    plt.savefig("graphs/train_boxplot.svg")

    plt.figure(figsize = (10, 5))
    sns.barplot(data = df, x = "penalty", y = "final_acc", capsize=.2)
    plt.xlabel("Loss Parameter Φ")
    plt.ylim([0.8, 1.0])
    plt.ylabel("Validation Accuracy")
    plt.savefig("graphs/val_boxplot.svg")


def load_and_plot_all(metric_files, save = False):
    print("Loading from", metric_files, "with Save = ", save)
    df = files_to_df(metric_files)

    print("Plotting Train and Val Curves...")
    plot_train_val_curves(df, save = save)

    print("Plotting Performance Boxplots...")
    plot_boxplots(df, save = save)

def inhibition_plot(modelpath, modeltype='mnist', n=2, strength=0.2, cocoroot='', annpath='', metadatapath='', save = False):
    net, runner, test_loader = load_model_and_data(modelpath, n = n, strength = strength, modeltype = modeltype,\
                                               cocoroot = cocoroot, annpath = annpath, metadatapath = metadatapath)
    
    loader = DataLoader(test_loader.dataset, batch_size = 1, shuffle = True, num_workers=4)
    inhibiteds = []
    not_inhibiteds = []
    
    print("Processing. Please Wait...")
    for i, (x, data, labels) in enumerate(loader):
        masks, hiddens, ior, selects_x = runner.visualize(x, data, labels)
        x = x.detach().cpu().numpy()
        
        inhibited = 0
        not_inhibited = 0
        
        for k in range(n):
            mask = masks[k]
            hidden = hiddens[k]
            selectx = selects_x[k]
            
            targethidden = ((x - selectx + 0.000001) / (x + 0.000001)) / strength
            targethidden = (targethidden > 0.5)
            
            inhibited += np.mean(hidden[targethidden])
            not_inhibited += np.mean(hidden[np.logical_not(targethidden)])
        
        inhibiteds.append(inhibited / n)
        not_inhibiteds.append(not_inhibited / n)
        
        if i % int(len(loader) / 30) == 0: 
            print(int(i*100/len(loader)), "%")

    values = inhibiteds + not_inhibiteds
    labels = ["Not Attended"]*len(inhibiteds) + ["Attended"]*len(not_inhibiteds)
    df = pd.DataFrame({"inhibition": values, "condition": labels})

    colors = ["#d44e4e", "#0e4db3"]
    sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(10, 5))
    sns.set_style("white")
    sns.set_context("poster")
    sns.histplot(data = df, x = "inhibition", hue="condition")
    plt.ylabel("Frequency")
    plt.xlabel("Attention Gating (%)")
    plt.legend(["Attended", "Not Attended"], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
            
    if save: 
        plt.savefig("graphs/inhibition.svg")
    
    print("Mean Inhibition for Attended vs Not Attended")
    print(df.groupby("condition")[["inhibition"]].mean())
    print("SD Inhibition for Attended vs Not Attended")
    print(df.groupby("condition")[["inhibition"]].std())
    stat, p = ttest_rel(inhibiteds, not_inhibiteds)
    print("\tT Statistic %s\tP Value: %s"%(stat, p))


