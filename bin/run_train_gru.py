import argparse

import torch

from training.train_gru import train_driver
from utilities.file_utils import check_rm_dirs_like
from utilities.file_utils import load_config_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Configuration file with training parameters")
    parser.add_argument(
        "model_output_dir",
        type=str,
        help="Directory containing the artifacts of the trained model",
    )
    parser.add_argument(
        "dataset_file", type=str, help="File containing all the training annotations"
    )

    args = parser.parse_args()
    config_fname = args.config
    model_output_dir = args.model_output_dir
    dataset_file = args.dataset_file

    check_rm_dirs_like(model_output_dir)

    params = load_config_params(config_fname)
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_driver(dataset_file, model_output_dir, params)
