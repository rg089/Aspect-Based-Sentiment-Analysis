import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_json(fname, data):
    """
    Save the output to a file
    """
    with open(fname, 'w') as f:
        json.dump(data, f)


def start_table():
    print("-"*66)
    print(f"|{'EPOCH':^15}|{'BATCH':^15}|{'TRAIN LOSS':^16}|{'EVAL LOSS':^15}|")
    print("-"*66)


def flatten(l):
    """
    l is a numpy array of dim > 1
    """
    if len(l.shape) == 2:
        l = [item for sublist in l for item in sublist]
        return l

    elif len(l.shape) == 3:
        l = l.argmax(axis=-1)
        return flatten(l)


def plot_loss(loss, title, xlabel, ylabel, save_path=None):
    sns.set(style="darkgrid")
    sns.lineplot(x=np.arange(len(loss)), y=loss).set(title=title, xlabel=xlabel, ylabel=ylabel)
    if save_path is not None:
        plt.savefig(save_path)


def extract_aspects(tokens, output):
    mapper = {idx: word for idx, word in enumerate(["Q", "B", "I", "O", "X"])}
    predicted = [mapper[pred] for pred in output]
    
    aspects = []
    current = ""
    
    for token, pred_tag in zip(tokens, predicted):
        if pred_tag == "B":
            if current != "":
                aspects.append(current)
            current = token
            
        elif pred_tag == "I":
            current += " " + token
            
        elif pred_tag == "X":
            if token.startswith("##"): current += token[2:]
            else: current += token
                
        else:
            if current != "":
                aspects.append(current)
                current = ""

    if current != "":
        aspects.append(current)
        
    return aspects