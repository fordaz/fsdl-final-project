import os
import json
import shutil
from argparse import Namespace
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def load_json(full_fname):
    with open(full_fname, "r") as input_file:
        raw_json = input_file.read()
        return json.loads(raw_json)


def load_json(base_dir, fname):
    return load_json(os.path.join(base_dir, fname))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def check_rm_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def save_by_line(body, base_output_dir, out_fname):
    with open(os.path.join(base_output_dir, out_fname), "w") as out_file:
        for line in body:
            out_file.write(line + "\n")


def load_yaml(full_fname):
    with open(full_fname, "r") as input_file:
        raw_yaml = input_file.read()
        return load(raw_yaml, Loader=Loader)


def load_config_params(full_fname):
    params = load_yaml(full_fname)
    return Namespace(**params)