import os
import json
import logging
import pandas as pd
from pathlib import Path

from utilities.file_utils import load_json, save_by_line

def generate_block_sentences(input_json):
    """
    Extracting each non duplicate annotation block as a json string one per line.
    """
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

def remove_json_keys(input_json):
    """
    Most of the text within the words collection is 100% the same as the text of the block.
    Removing it should not reduce the quality of the data and should reduce the length of the sequence.
    """
    for block in input_json["form"]:
        block.pop("words", None)
    return input_json


def wrap_annotations(clean_annotations, split):
    return [{"annotation": annotation, "split": split} for annotation in clean_annotations]


def process_raw_annotations(file_names, split_tagger):
    data_set = []
    for i, fname in enumerate(file_names):
        # getting the split tag based on index
        split = split_tagger(i)

        raw_annonations = load_json(fname)

        # remove certain metadata from the annotations
        remove_json_keys(raw_annonations)

        # generate single line annotation sentences
        clean_annotations = generate_block_sentences(raw_annonations)

        # wrap them into a dict to serialize as data frame
        wrapped_annotations = wrap_annotations(clean_annotations, split)

        data_set.extend(wrapped_annotations)
    return data_set


def generate_clean_annotations(input_train_dir,
                               input_test_dir,
                               train_split,
                               full_output_fname):
    p = Path(input_train_dir)
    training_files = list(p.glob('**/*.json'))

    num_files = len(training_files)
    num_training_files = int(train_split*num_files)

    full_dataset = []

    train_val_dataset = process_raw_annotations(training_files,
                                                 lambda idx: "train" if idx <= num_training_files else "val")
    full_dataset.extend(train_val_dataset)

    p = Path(input_test_dir)
    testing_files = list(p.glob('**/*.json'))

    test_dataset = process_raw_annotations(testing_files,
                                            lambda idx: "test")
    full_dataset.extend(test_dataset)

    full_dataset_df = pd.DataFrame(full_dataset)
    print(full_dataset_df.split.value_counts())

    full_dataset_df.to_csv(full_output_fname, columns = ["annotation", "split"], index=False)
