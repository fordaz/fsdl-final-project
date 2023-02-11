import boto3
import pandas as pd
import json

from tools.invoker_utils import *


def handle_request_response(app_name, smrt, base_directory):
    test_data = pd.DataFrame(
        {
            "num_pages": [1],
            "min_num_annot": [30],
            "max_num_annot": [50],
            "max_annot_length": [100],
            "temperature": [1.0],
        }
    )

    input_data = test_data.to_json(orient="split")
    prediction = smrt.invoke_endpoint(
        EndpointName=app_name,
        Body=input_data,
        ContentType="application/json; format=pandas-split",
    )

    if prediction.get("ResponseMetadata", {}).get("HTTPStatusCode", None) == 200:
        raw_body = prediction["Body"]
        body = raw_body.read().decode("ascii")
        save_syn_annotations_kits(base_directory, json.loads(body))
    else:
        print(f"Could not get a successful reponse {prediction}")


if __name__ == "__main__":
    base_directory = "synthetic"

    app_name = "fsdl-synth-docs"
    region = "us-west-2"

    sm = boto3.client("sagemaker", region_name=region)
    smrt = boto3.client("runtime.sagemaker", region_name=region)

    endpoint = sm.describe_endpoint(EndpointName=app_name)
    if endpoint.get("EndpointStatus", "") == "InService":
        handle_request_response(app_name, smrt, base_directory)
    else:
        print(f"Could not get inference. Endpoint status: {endpoint}")
