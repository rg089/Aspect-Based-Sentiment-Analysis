import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from utils import load_data, start_table, flatten, plot_loss, extract_aspects
import argparse, os


class BERT_LSTM(nn.Module):
    def __init__(self, bert_path, hidden_dim, num_layers, dropout, num_classes, bidirectional=False, freeze_bert=False):
        super(BERT_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert = AutoModel.from_pretrained(bert_path).to(self.device)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, num_classes)


        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, tokenized, h=None, c=None):
        batch_size = tokenized['input_ids'].size(0)
        if h is None:
            h, c = self.init_hidden(batch_size)

        outputs = self.bert(**tokenized)
        hidden_outputs, pooled_outputs = outputs[0], outputs[1]
        out, (h, c) = self.lstm(hidden_outputs, (h, c))
        out = self.fc(out)
        return out, (h, c)
        
    def init_hidden(self, batch_size):
        num_l = self.num_layers * 2 if self.bidirectional else self.num_layers
        hidden = torch.zeros(num_l, batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(num_l, batch_size, self.hidden_dim).to(self.device)
        return hidden, cell


def mask(pred, tags):
    tags = np.array(tags); pred = np.array(pred)
    assert len(pred) == len(tags)
    mask = tags != -100
    tags = tags[mask]; pred = pred[mask]
    assert len(pred) == len(tags)
    return pred, tags


def compute_metrics(pred, tags):
    pred = flatten(pred); tags = flatten(tags)
    pred, tags = mask(pred, tags)
    accuracy = (pred == tags).mean()
    print(classification_report(tags, pred, target_names=['B', 'I', 'O', 'X']))
    return accuracy


def train_epoch(model, optimizer, criterion, train_loader, val_loader, device, epoch_num, plot_dir, print_every=50):
    total_loss = 0
    batch_losses = []
    eval_losses = []

    for idx, sample in enumerate(train_loader):
        model.train()

        text = sample['text']
        tokenized = tokenizer(text, padding="max_length", max_length=75, truncation=True, return_tensors="pt").to(device)
        tags = sample['tags']

        optimizer.zero_grad()
        output, (_, _) = model(tokenized)
        output = output.permute(0, 2, 1) # Changing to (batch_size, num_classes, seq_len) as expected by cross_entropy
        loss = criterion(output, tags)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_losses.append(loss.item())

        if (idx+1) % print_every == 0:
            eval_loss = evaluate(model, criterion, val_loader, device)
            eval_losses.append(eval_loss)
            print(f"|{epoch_num:^15}|{idx+1:^15}|{loss.item():^16.4f}|{eval_loss:^15.4f}|")

        plot_loss(batch_losses, "Training Loss", "Step", "Loss", plot_dir+ "/batch_loss.png")

    epoch_loss =  total_loss / len(train_loader)
    print("-"*66 + "\n")
    return model, epoch_loss, batch_losses, eval_losses


def evaluate(model, criterion, test_loader, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            text = sample['text']
            tokenized = tokenizer(text, padding="max_length", max_length=75, truncation=True, return_tensors="pt").to(device)
            tags = sample['tags']

            output, (_, _) = model(tokenized)
            output = output.permute(0, 2, 1) # Changing to (batch_size, num_classes, seq_len) as expected by cross_entropy
            loss = criterion(output, tags)

            total_loss += loss.item()

    epoch_loss =  total_loss / len(test_loader)
    return epoch_loss


def evaluate_complete(model, test_loader, device):
    model.eval()

    outputs = None
    tags = None

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            text = sample['text']
            tags_ = sample['tags'].cpu().numpy()

            tokenized = tokenizer(text, padding="max_length", max_length=75, truncation=True, return_tensors="pt").to(device)

            output, (_, _) = model(tokenized)
            output= output.cpu().numpy()

            # Concatenate to the outputs
            outputs = np.concatenate((outputs, output), axis=0) if outputs is not None else output
            tags = np.concatenate((tags, tags_), axis=0) if tags is not None else tags_

    accuracy = compute_metrics(outputs, tags)
    return accuracy


def predict_sample(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(text, padding="max_length", max_length=75, truncation=True, return_tensors="pt").to(device)
        output, (_, _) = model(tokenized)
        output = output.argmax(dim=-1).cpu()[0].tolist()
        
        tokens = tokenizer.tokenize(text, add_special_tokens=True, max_length=75, padding='max_length', truncation=True)
        idx = tokens.index("[SEP]")
        
        output = output[1:idx]
        tokens = tokens[1:idx]
        
        aspects = extract_aspects(tokens, output)
        return aspects


def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, plot_dir, print_every=20):
    epoch_losses = []
    batch_losses = []
    eval_losses = []

    for epoch in range(epochs):
        # print(f"[INFO] STARTING EPOCH {epoch+1}:\n")
        start_table()
        model, epoch_loss, batch_losses_, eval_losses = train_epoch(model, optimizer, criterion, train_loader, val_loader, device, epoch+1, plot_dir, print_every)

        epoch_losses.append(epoch_loss)
        batch_losses.extend(batch_losses_)
        eval_losses.extend(eval_losses)

        print(f"EPOCH: {epoch+1} AVG LOSS: {epoch_loss}\n")
        evaluate_complete(model, val_loader, device)

    plt.figure()  
    plot_loss(batch_losses, "Training Loss", "Step", "Loss", plot_dir+"/batch_loss.png")
    
    plt.figure()
    plot_loss(epoch_losses, "Training Loss", "Epoch", "Loss", plot_dir+"/epoch_loss.png")
    
    return model, epoch_losses, batch_losses, eval_losses



parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str, default="bert-base-cased", help="Path to BERT model")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of LSTM")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--freeze_bert", type=bool, default=True, help="Freeze BERT weights")
parser.add_argument("--num_layers", type=int, default=2, help="Number of Layers in LSTM")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--save_path", type=str, default="models/BERT_LSTM.pt", help="Path to save model")
parser.add_argument("--plot_dir", type=str, default="plots/train_ae", help="Directory to save the plots")
parser.add_argument("--log_every", type=int, default=10, help="Log every n batches")
parser.add_argument("--dataset", type=str, default="laptops", help="Dataset to train on")

args = parser.parse_args()

# PARAMS
bert_path = args.bert_path
hidden_dim = args.hidden_dim
num_layers = args.num_layers
dropout = args.dropout
num_classes = 5
bidirectional = True
freeze_bert = args.freeze_bert
if freeze_bert: print("[INFO] BERT has been frozen!\n")
batch_size = args.batch_size
shuffle = True
dataset = args.dataset
epochs = args.epochs
print_every = args.log_every
plot_dir = args.plot_dir + "_" + dataset
save_path = args.save_path.split(".pt")[0] + "_" + dataset + ".pt"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

train_loader = load_data(f"data/train_{dataset}.json", batch_size=batch_size, shuffle=shuffle)
val_loader = load_data(f"data/test_{dataset}.json", batch_size=batch_size, shuffle=shuffle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(bert_path)
model = BERT_LSTM(bert_path, hidden_dim, num_layers, dropout, num_classes, bidirectional, freeze_bert).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(model.parameters(), lr=args.lr)


model, epoch_losses, batch_losses, eval_losses = train(model, 
                        optimizer, criterion, epochs, train_loader, val_loader, device, plot_dir, print_every)

torch.save(model, save_path)


""" 
To train, use the following command:
python train_ae.py --bert_path bert-base-cased --hidden_dim 128 --epochs 10 --batch_size 16 --freeze_bert True --num_layers 2 --dropout 0.1 --lr 5e-5 --save_path models/BERT_LSTM.pt --plot_dir plots/train_ae --dataset laptops
"""
