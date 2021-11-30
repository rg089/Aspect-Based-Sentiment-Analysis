import torch
from torch.utils.data import Dataset, DataLoader
import json
from .tag_to_idx import load_mapper


class ABSA_Dataset(Dataset):
    def __init__(self, data_path, mapper_path):
        self.data_path = data_path
        self.data = json.load(open(data_path, "r"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(mapper_path)
        self.mapper = load_mapper(mapper_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        item["text"] = self.data[idx]["text"]
        if "tags" in self.data[idx]:
            item["tags"] = [self.mapper[tag] for tag in self.data[idx]["tags"]]
            item["tags"] = torch.tensor(item["tags"], dtype=torch.long, device=self.device)
        if "aspect" in self.data[idx]:
            item["aspect"] = self.data[idx]["aspect"]
        if "polarity" in self.data[idx]:
            item["polarity"] = self.mapper[self.data[idx]["polarity"]]
        # item["tokens"] = self.data[idx]["tokens"]
        return item


def load_data(data_path, batch_size=16, shuffle=True, mapper_path="models/tag2idx.json"):
    dataset = ABSA_Dataset(data_path, mapper_path=mapper_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader
