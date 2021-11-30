from transformers import AutoTokenizer
import argparse

from utils import save_json
from preprocessing.convert_format import read_file, parse_tree
from preprocessing.tag_generator import generate_tags



def process_sc(input_file, output_file):
    """
    Processes a file for sentiment classification and saves the result in a file.

    :param input_file: the input file
    :param output_file: the output file
    :param tokenizer: the tokenizer
    :param max_len: the maximum length of a sentence
    :return: None
    """
    root = read_file(input_file)
    parsed = parse_tree(root)
    data = []
    for sample in parsed:
        for aspect in sample['aspects']:
            new_sample = {}
            new_sample['text'] = sample['text']
            new_sample['aspect'] = aspect['term']
            new_sample['polarity'] = aspect['polarity']
            data.append(new_sample)

    save_json(output_file, data)


def process_ae(input_file, output_file, tokenizer, max_length=75):
    """
    Processes a file for aspect extraction and saves the result in a file.

    :param input_file: the input file
    :param output_file: the output file
    :param tokenizer: the tokenizer
    :param max_len: the maximum length of a sentence
    :return: None
    """
    root = read_file(input_file)
    data = parse_tree(root)
    for sample in data:
        tags, tokens, _ = generate_tags(sample, tokenizer, max_length)
        sample['tags'] = tags
        sample['tokens'] = tokens
    save_json(output_file, data)


if __name__ == "__main__":
    # Instantiate BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    # Get the input and output files
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', "-i", type=str, default='input.xml', help='input file')
    parser.add_argument('--output_file', "-o", type=str, default='output.json', help='output file')
    parser.add_argument('--max_length', "-l", type=int, default=75, help='maximum length of a sentence')
    parser.add_argument('--task', "-t", type=str, default='ae', help='task')

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    max_length = args.max_length
    task = args.task

    if task == 'ae':
        process_ae(input_file, output_file, tokenizer, max_length=max_length)
    else:
        process_sc(input_file, output_file)
