import os
import json

def load_json(base_dir, fname):
    with open(os.path.join(base_dir, fname), "r") as input_file:
        raw_json = input_file.read()
        return json.loads(raw_json)


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_by_line(body, base_output_dir, out_fname):
    with open(os.path.join(base_output_dir, out_fname), "w") as out_file:
        for line in body:
            out_file.write(line + "\n")
