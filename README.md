# Overview

This is the repo of my [final project](http://fordaz.github.io/synthetic-annotated-documents/) for the [Full Stack Deep Learning 2021](https://twitter.com/full_stack_dl) Course organized and taught by Sergey Karayev and Josh Tobin (Spring 2021).

# Project structure

Here is a brief description of the project structure:

## Core ML

* preprocessing: basic python pre-processing logic to generate single-line JSON annotations.
* models: python code for the model architecture, dataset, vectorizer and vocabulary.
* serving: python code for the model wrapper.
* training: python code with the training loop.

## Infra related folders

* ansible: ansible playbook to automate the training steps on the remote machine.
* terraform: terraform files to provision the infrastructure.
* scripts: auxility shell scripts to automate repetitive tasks.
* infra_state: text files storing EC2 instance id and DNS name to be used in the scripts.

## Datasets and artifacts

* datasets
    * funsd: the original funsd dataset
    * generated: the pre-processed annotations.
* model_artifacts: all the trained models, storing only DVC references, actual artifacts stored on S3.
* synthetic: generated synthetic data, storing only DVC references, actual artifacts stored on S3.

## Auxiliary Tools

* bin: driver scripts for launching the training, data pre-processing, wrapper creation.
* config: yaml files with training hyper-parameters for local and remote training.
* tools: python code to invoke the model locally and remotely.
* utilities: python code for file and image utilities.
* presentation: images and markdown presentation of the project.

# Pre-requisites

* Install terraform
* Install ansible
* AWS cli install and configured
* Install DVC, MLFlow and configure it according to your needs.
* Have github installed and configured
* Use conda or another package manager and install dependencies using requirements-local.txt file

# Pre-commit hooks

Few pre-commit hooks are in place.

## Working locally

* Include the project root directory in the `PYTHONPATH` so all the modules are accessible for import.

```
export PYTHONPATH=./fsdl-final-project:$PYTHONPATH
```

* Configure training parameters located at `./config/train_local.yaml`
* Training the model

```
./scripts/train_gru_model_local.sh
```

* Create model wrapper

```
./scripts/create_gru_model_wrapper.sh
```

* Run the model wrapper locally

```
./scripts/serve_gru_model_local.sh
```

* Generate some synthetic data.

```
./scripts/inference_local.sh
```

## Remote training

* Provision infrastructure

```
terraform apply
```

* Configure training parameters located at `./config/train_aws.yaml`
* Training the model remotely

```
ansible-playbook train_gru_remote.yaml --ask-vault-pass
```

* Pull trained model artifacts
```
git pull
```

* Create model wrapper

```
./scripts/create_gru_model_wrapper.sh
```

## Build AWS Sagemaker image

```
./scripts/sagemaker_build.sh
```

## Test AWS Sagemaker image locally

```
./scripts/sagemaker_deploy_local.sh
./scripts/sagemaker_inference_local.sh
```

## Test AWS Sagemaker image remotely

```
./scripts/sagemaker_deploy_remote.sh
./scripts/sagemaker_inference_remote.sh
```
