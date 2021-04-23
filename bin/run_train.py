import argparse

from training.train import train_driver
from utilities.file_utils import check_rm_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_output_dir', type=str, help='Directory containing the artifacts of the trained model')
    parser.add_argument('dataset_file', type=str, help='File containing all the training annotations')

    args = parser.parse_args()
    model_output_dir = args.model_output_dir
    dataset_file = args.dataset_file

    check_rm_dir(model_output_dir)
    check_rm_dir(f"{model_output_dir}_wrapper")

    train_driver(dataset_file, model_output_dir)