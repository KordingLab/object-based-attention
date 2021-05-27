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
    plt.xlabel("Gating Loss Penalty")
    plt.ylabel("Training Accuracy")
    plt.ylim([0.8, 1.0])
    plt.savefig("graphs/train_boxplot.svg")

    plt.figure(figsize = (10, 5))
    sns.barplot(data = df, x = "penalty", y = "final_acc", capsize=.2)
    plt.xlabel("Gating Loss Penalty")
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

    


