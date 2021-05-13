import requests
import json

from tools.invoker_utils import *

def handle_request_response(model_endpoint, base_directory):

    headers = {'Content-Type': 'application/json'}

    inference_request = {
        "columns": ["num_pages", "min_num_annot", "max_num_annot", "max_annot_length", "temperature"],
        "data": [[1, 20, 30, 100, 1.0]]
    }

    response = requests.post(model_endpoint, data=json.dumps(inference_request), headers=headers)

    if response.status_code == 200:
        save_syn_annotations_kits(base_directory, response.json())
    else:
        print(f"Could not get a successful reponse {dir(response)}")


if __name__ == "__main__":
    model_endpoint = "http://127.0.0.1:5001/invocations"
    base_directory = "synthetic"

    handle_request_response(model_endpoint, base_directory)