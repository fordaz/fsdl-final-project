from training.train import train_driver
from utilities.file_utils import check_rm_dir

save_model_dir = "saved_model_001"
dataset = "./training/test_input_annotations.txt"

check_rm_dir(save_model_dir)
check_rm_dir(f"{save_model_dir}_wrapper")

train_driver(dataset, save_model_dir)