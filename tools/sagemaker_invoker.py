import boto3
import pandas as pd

app_name = 'fsdl-synth-docs'
region = 'us-west-2'

sm = boto3.client('sagemaker', region_name=region)
smrt = boto3.client('runtime.sagemaker', region_name=region)

# Check endpoint status
endpoint = sm.describe_endpoint(EndpointName=app_name)
print("Endpoint status: ", endpoint["EndpointStatus"])    


test_data = pd.DataFrame(
    {
        "input": ["abc"]
    }
)

input_data = test_data.to_json(orient="split")
prediction = smrt.invoke_endpoint(
    EndpointName=app_name,
    Body=input_data,
    ContentType='application/json; format=pandas-split'
)
raw_prediction = prediction['Body']
print(f"Full prediction response {raw_prediction}")
prediction = raw_prediction.read().decode("ascii")
print(prediction)