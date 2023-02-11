import argparse
import logging
import os

from preprocessing.annotations import generate_clean_annotations
from utilities.file_utils import check_mkdir

logging.basicConfig(filename="preprocessing.log", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_train_dir",
        type=str,
        help="Input directory with raw training annotations dataset",
    )
    parser.add_argument(
        "input_test_dir",
        type=str,
        help="Input directory with raw test annotations dataset",
    )
    parser.add_argument(
        "train_split",
        type=float,
        help="Percentage of training annotations files for training, the rest is for validation",
    )
    parser.add_argument(
        "output_dataset_dir",
        type=str,
        help="Output directory for pre-processed annotations dataset",
    )
    parser.add_argument(
        "output_dataset_file_name",
        type=str,
        help="Output file name for pre-processed annotations dataset",
    )

    args = parser.parse_args()

    full_output_fname = os.path.join(args.output_dataset_dir, args.output_dataset_file_name)

    logging.info(
        f"Using input folders {args.input_train_dir} {args.input_test_dir}, output file {full_output_fname}"
    )

    # Check and create output dir if needed
    check_mkdir(args.output_dataset_dir)

    generate_clean_annotations(
        args.input_train_dir, args.input_test_dir, args.train_split, full_output_fname
    )
