import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import json, os
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, start_table, sentiment_mapper, plot_loss
import argparse


class BERT_CNN(nn.Module):
    def __init__(self, bert_path, num_classes, filter_size, num_filters, seq_len=75, freeze_bert=False):
        super(BERT_CNN, self).__init__()
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert = AutoModel.from_pretrained(bert_path).to(self.device)
        self.cnn = nn.Conv2d(1, num_filters, kernel_size=(filter_size, self.bert.config.hidden_size), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=(seq_len-filter_size+1, 1))
        self.fc = nn.Linear(num_filters, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, tokenized, h=None, c=None):
        # batch_size = tokenized['input_ids'].size(0)
        outputs = self.bert(**tokenized)
        hidden_outputs, pooled_outputs = outputs[0], outputs[1]
        out = self.cnn(hidden_outputs.unsqueeze(1))
        out = F.relu(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1) # Flattening to (batch_size, num_filters)
        logits = self.fc(out)
        return logits


def compute_metrics(pred, tags):
    if len(pred.shape) == 2:
        pred = pred.argmax(axis=-1)
    accuracy = (pred == tags).mean()
    print(classification_report(tags, pred, target_names=["positive", "neutral", "negative", "conflict"]))
    return accuracy


def train_epoch(model, optimizer, criterion, train_loader, val_loader, device, epoch_num, print_every=50):
    total_loss = 0
    batch_losses = []
    eval_losses = []

    for idx, sample in enumerate(train_loader):
        model.train()

        text = sample['text']
        aspect = sample['aspect']
        y = sample['polarity'].to(device)

        tokenized = tokenizer(text, aspect, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt").to(device)

        optimizer.zero_grad()
        output = model(tokenized)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_losses.append(loss.item())

        if (idx+1) % print_every == 0:
            # print(f"EPOCH: {epoch_num} BATCH:{idx+1}/{len(train_loader)} LOSS: {loss.item()}")
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
            aspect = sample['aspect']
            y = sample['polarity'].to(device)

            tokenized = tokenizer(text, aspect, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt").to(device)

            output = model(tokenized)
            loss = criterion(output, y)

            total_loss += loss.item()

    epoch_loss =  total_loss / len(test_loader)
    # print(f"\VAL LOSS: {epoch_loss}\n")
    return epoch_loss


def evaluate_complete(model, test_loader, device):
    model.eval()

    outputs = None
    y = None

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            text = sample['text']
            aspect = sample['aspect']
            y_ = sample['polarity'].cpu().numpy()

            tokenized = tokenizer(text, aspect, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt").to(device)

            output = model(tokenized).cpu().numpy()

            # Concatenate to the outputs
            outputs = np.concatenate((outputs, output), axis=0) if outputs is not None else output
            y = np.concatenate((y, y_), axis=0) if y is not None else y_

    accuracy = compute_metrics(outputs, y)
    return accuracy


def predict_sample(text, aspect, model, tokenizer):
    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(text, aspect, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt").to(device)
        output = model(tokenized)
        output = output.argmax(dim=-1).cpu()[0].item()
        
        mapper = json.load(open(mapper_path, "r"))
        reverse_mapper = {v: k for k, v in mapper.items()}

        sentiment = reverse_mapper[output]
        return sentiment


def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, print_every=50):
    epoch_losses = []
    batch_losses = []
    eval_losses = []

    for epoch in range(epochs):
        # print(f"[INFO] STARTING EPOCH {epoch+1}:\n")
        start_table()
        model, epoch_loss, batch_losses_, eval_losses = train_epoch(model, optimizer, criterion, train_loader, val_loader, device, epoch+1, print_every)

        epoch_losses.append(epoch_loss)
        batch_losses.extend(batch_losses_)
        eval_losses.extend(eval_losses)

        print(f"EPOCH: {epoch+1} AVG LOSS: {epoch_loss}\n")
        evaluate_complete(model, val_loader, device)

    plt.figure()  
    plot_loss(batch_losses, "Training Loss", "Step", "Loss")
    
    plt.figure()
    plot_loss(epoch_losses, "Training Loss", "Epoch", "Loss")
    
    return model, epoch_losses, batch_losses, eval_losses


parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str, default="bert-base-cased", help="Path to BERT model")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--freeze_bert", type=bool, default=False, help="Freeze BERT weights")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--save_path", type=str, default="models/BERT_CNN.pt", help="Path to save model")
parser.add_argument("--plot_dir", type=str, default="plots/train_sc", help="Directory to save the plots")
parser.add_argument("--log_every", type=int, default=10, help="Log every n batches")
parser.add_argument("--dataset", type=str, default="laptops", help="Dataset to train on")
parser.add_argument("--num_filters", type=int, default=64, help="Number of filters in the CNN")
parser.add_argument("--filter_size", type=int, default=10, help="Filter size of the CNN")
parser.add_argument("--seq_len", type=int, default=75, help="Sequence length")

args = parser.parse_args()

# PARAMS
bert_path = args.bert_path
num_filters = args.num_filters
filter_size = args.filter_size
num_classes = 4
freeze_bert = args.freeze_bert
if freeze_bert: print("[INFO] BERT has been frozen!\n")
batch_size = args.batch_size
shuffle = True
dataset = args.dataset
epochs = args.epochs
print_every = args.log_every
plot_dir = args.plot_dir + "_" + dataset
save_path = args.save_path.split(".pt")[0] + "_" + dataset + ".pt"
seq_len = args.seq_len

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

mapper_path = "models/sent2idx.json"

train_loader = load_data(f"data/train_{dataset}_sc.json", batch_size=batch_size, shuffle=shuffle, mapper_path=mapper_path)
val_loader = load_data(f"data/test_{dataset}_sc.json", batch_size=batch_size, shuffle=shuffle, mapper_path=mapper_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(bert_path)
model = BERT_CNN(bert_path, num_classes, filter_size, num_filters, seq_len, freeze_bert).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=args.lr)


model, epoch_losses, batch_losses, eval_losses = train(model, 
                        optimizer, criterion, epochs, train_loader, val_loader, device, print_every)

torch.save(model, save_path)

"""
To train, use the following command:
python train_sc.py --bert_path bert-base-cased --epochs 15 --batch_size 16 --lr 5e-5 --save_path models/BERT_CNN.pt --plot_dir plots/train_sc --log_every 10 --dataset laptops --num_filters 64 --filter_size 10 --seq_len 75    
"""


