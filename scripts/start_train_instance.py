#!/usr/bin/env python

from pathlib import PurePath

import hcl
import boto3


def load_variables(base_dir):
    vars_path = PurePath(base_dir, 'variables.tf')
    with open(vars_path, 'r') as fp:
        obj = hcl.load(fp)
        return obj['variable']


def get_instance_id(ec2_client, tf_vars):
    assert tf_vars['ami_id']['default'], "No ami_id in configuration file"
    ami_id = tf_vars['ami_id']['default']
    response = ec2_client.describe_instances(
            Filters=[{'Name': 'image-id', 'Values': ['ami-01fe82fb0c26ae580',]},],
            MaxResults=5)
    assert len(response['Reservations']) == 1, "Non unique instance id returned"
    assert len(response['Reservations'][0]['Instances']) == 1, "Non unique instance id returned"
    return response['Reservations'][0]['Instances'][0]['InstanceId']



def start_instance(ec2_resource, instance_id):
    instance = ec2_resource.Instance(instance_id)
    response = instance.start()
    print(response)
    assert len(response['StartingInstances']) == 1, "Expected only one instance to be starting"
    current_state = response['StartingInstances'][0]['CurrentState']['Name']
    print(f"Instace {instance_id} in current state {current_state}")

# TODO update ansible host file after getting a new public DNS endpoint

if __name__ == '__main__':
    base_dir = "./terraform"

    ec2_client = boto3.client('ec2')
    ec2_resource = boto3.resource('ec2')

    tf_vars = load_variables(base_dir)

    instance_id = get_instance_id(ec2_client, tf_vars)

    start_instance(ec2_resource, instance_id)
