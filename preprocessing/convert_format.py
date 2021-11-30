# Preprocess the ABSA dataset

import xml.etree.ElementTree as ET
import json

# output format: [{'id: '', 'text': '', 'aspects': [{'term': '', 'from': 0, 'to': 0, 'polarity': ''}]}]

def read_file(fname):
    """
    Read the xml file and return the root element
    """
    tree = ET.parse(fname)
    return tree.getroot()

def parse_tree(root):
    """
    Parse the xml tree and return the output in required format
    """
    data = []
    for sentence in root.iter('sentence'):
        sentence_info = {}
        sentence_info['id'] = sentence.attrib['id']

        sentence_info['text'] = sentence.find('text').text
        sentence_info['aspects'] = []

        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is None:
            continue

        for aspect in sentence.find('aspectTerms').findall('aspectTerm'):
            aspect_info = {}
            aspect_info['term'] = aspect.attrib['term']
            aspect_info['from'] = int(aspect.attrib['from'])
            aspect_info['to'] = int(aspect.attrib['to'])
            aspect_info['polarity'] = aspect.attrib['polarity']
            sentence_info['aspects'].append(aspect_info)

        data.append(sentence_info)

    print(f"[INFO] Processing Complete! {len(data)} Samples Processed!")
    return data
