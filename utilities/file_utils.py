import json
import os
import shutil
from argparse import Namespace
import yaml

# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
from pathlib import Path


def load_json(full_fname):
    with open(full_fname, "r") as input_file:
        raw_json = input_file.read()
        return json.loads(raw_json)


def save_json(body, out_fname):
    with open(out_fname, "w") as out_file:
        out_file.write(json.dumps(body))


# def load_json(base_dir, fname):
#     return load_json(os.path.join(base_dir, fname))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def check_rm_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def check_rm_dirs_like(dir_name):
    path = Path(dir_name)
    path_parent = path.parent
    for dir_like in path_parent.glob(f"*{path.parts[-1]}*"):
        if os.path.exists(dir_like) and os.path.isdir(dir_like):
            shutil.rmtree(dir_like)


def save_by_line(body, base_output_dir, out_fname):
    with open(os.path.join(base_output_dir, out_fname), "w") as out_file:
        for line in body:
            out_file.write(line + "\n")


def load_yaml(full_fname):
    with open(full_fname, "r") as input_file:
        raw_yaml = input_file.read()
        # return load(raw_yaml, Loader=Loader)
        return yaml.safe_load(raw_yaml)


def load_config_params(full_fname):
    params = load_yaml(full_fname)
    return Namespace(**params)
