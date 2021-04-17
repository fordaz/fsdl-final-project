import os
import logging
import yaml
import argparse

from preprocessing.annotations import generate_clean_annotations
from utilities.file_utils import check_mkdir


logging.basicConfig(filename='preprocessing.log', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('base_input_dir', type=str, help='Directory containing all annotation files')
    parser.add_argument('base_output_dir', type=str, help='Directory to contain all cleaned annotation files')

    args = parser.parse_args()
    base_input_dir = args.base_input_dir
    base_output_dir = args.base_output_dir

    params = yaml.safe_load(open('params.yaml'))['preprocess']
    combined_fname = params['combined_fname']

    logging.info(f"Using input folder {base_input_dir}, output folder {base_output_dir} and combined file {combined_fname}")

    # Check and create output dir if needed
    check_mkdir(base_output_dir)

    generate_clean_annotations(base_input_dir, base_output_dir, combined_fname)