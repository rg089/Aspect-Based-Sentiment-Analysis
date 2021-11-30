import json

def sentiment_mapper(mapper_path="models/sent2idx.json"):
    """
    Loads the sentiment mapper from the specified file.
    """
    with open(mapper_path, "r") as f:
        return json.load(f)