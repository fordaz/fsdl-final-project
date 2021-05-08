from sys import version_info
import argparse
import cloudpickle

import mlflow

from models.gru_annotations_lm_wrapper import GRUAnnotationsWrapper
from utilities.file_utils import check_rm_dir

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

def save_model_wrapper(saved_model_fname, dataset_fname):
    artifacts = {
        "pytorch_model": saved_model_fname,
        "dataset_input_file": dataset_fname
    }

    conda_env = {
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            "pytorch=1.7.0",
            "torchvision=0.8.1",
            'pip',
            {
                'pip': [
                    'mlflow',
                    'cloudpickle=={}'.format(cloudpickle.__version__),
                ],
            },
        ],
        'name': 'mlflow-env-wrapper'
    }

    mlflow_pyfunc_model_fname = f"{saved_model_fname}_gru_wrapper"
    mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_fname, code_path=["training", "models"],
                             python_model=GRUAnnotationsWrapper(), 
                             artifacts=artifacts, conda_env=conda_env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str, help='Trained Model Directory containing the model artifacts')
    parser.add_argument('dataset_file', type=str, help='File containing all the training annotations')

    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_file = args.dataset_file

    check_rm_dir(f"{model_dir}_gru_wrapper")

    save_model_wrapper(model_dir, dataset_file)