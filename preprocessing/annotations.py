import os
import json
import logging

from utilities.file_utils import load_json, save_by_line

"""
Extracting each non duplicate annotation block as a json string one per line.
"""
def generate_block_sentences(input_json):
    blocks_lookup = { block["id"]: block for block in input_json["form"] }
    blocks = input_json["form"]
    sentences = []
    included = set()

    for block in blocks:
        if block["id"] in included:
            continue
        included.add(block["id"])
        sentence = json.dumps(block)
        sentences.append(sentence)

    return sentences

"""
Most of the text within the words collection is 100% the same as the text of the block.
Removing it should not reduce the quality of the data and should reduce the length of the sequence.
"""
def remove_json_keys(input_json):
    for block in input_json["form"]:
        block.pop("words", None)
    return input_json


transformations = [
    remove_json_keys, generate_block_sentences
]

def generate_clean_annotations(base_input_dir, base_output_dir, combined_fname, max_count=None):
    count = 0
    all_output_annotations = []

    for fname in os.listdir(base_input_dir):
        logging.info(f"Processing input file {fname}")

        # loading each annotations file
        input_transform = load_json(base_input_dir, fname)

        # preprocessing the file with multiple transformations
        for transformation in transformations:
            output_transform = transformation(input_transform)

        all_output_annotations.extend(output_transform)
        
        # saving a clean version of each each annotation file
        save_by_line(output_transform, base_output_dir, fname)

        count += 1
        if max_count and count >= max_count:
            break

    # saving a clean version of all the annotations, useful for training
    save_by_line(all_output_annotations, base_output_dir, combined_fname)