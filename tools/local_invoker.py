import requests
import json
import argparse

from tools.invoker_utils import *

def handle_request_response(model_endpoint, base_directory, num_pages, 
                            min_annotations, max_annotations, max_annotation_length, 
                            temperature):

    headers = {'Content-Type': 'application/json'}

    inference_request = {
        "columns": ["num_pages", "min_num_annot", "max_num_annot", "max_annot_length", "temperature"],
        "data": [[num_pages, min_annotations, max_annotations, max_annotation_length, temperature]]
    }

    response = requests.post(model_endpoint, data=json.dumps(inference_request), headers=headers)

    if response.status_code == 200:
        save_syn_annotations_kits(base_directory, response.json())
    else:
        print(f"Could not get a successful reponse {dir(response)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--port', type=int, help='Port number: mlflow 5001, sagemaker 5000', default=5001)
    parser.add_argument('--num_pages', type=int, help='Number of pages to generate', default=1)
    parser.add_argument('--min_annotations', type=int, help='Minimum number of annotations', default=20)
    parser.add_argument('--max_annotations', type=int, help='Maximum number of annotations', default=30)
    parser.add_argument('--max_annotation_length', type=int, help='Maximum number of annotations', default=100)
    parser.add_argument('--temperature', type=float, help='Maximum number of annotations', default=1.0)

    args = parser.parse_args()

    port = args.port
    num_pages = args.num_pages
    min_annotations = args.min_annotations
    max_annotations = args.max_annotations
    max_annotation_length = args.max_annotation_length
    temperature = args.temperature

    print(f"Running local inference on port {port}")

    model_endpoint = f"http://127.0.0.1:{port}/invocations"
    base_directory = "synthetic"

    handle_request_response(model_endpoint, base_directory, num_pages, 
                            min_annotations, max_annotations, max_annotation_length, 
                            temperature)