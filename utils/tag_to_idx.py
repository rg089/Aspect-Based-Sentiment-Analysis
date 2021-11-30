import json
import os


def create_mapper():
    tag_list = ["B", "I", "O", "X", "Q"]
    tag2idx = {tag: idx+1 for idx, tag in enumerate(tag_list)}
    # Since Q represents tags we don't want to compute loss over, set the index to -100.
    tag2idx["Q"] = -100

    return tag2idx


def save_mapper(filepath):
    tag2idx = create_mapper()
    with open(filepath, 'w') as f:
        json.dump(tag2idx, f)


def load_mapper(filepath='models/tag2idx.json'):
    if not os.path.exists(filepath):
        save_mapper(filepath)
    with open(filepath, 'r') as f:
        tag2idx = json.load(f)
    return tag2idx