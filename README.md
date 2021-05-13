# Overview

This repo of my final project for the [Full Stack Deep Learning 2021](https://twitter.com/full_stack_dl) Course organized and taught by Sergey Karayev and Josh Tobin.

# Project structure

Here is a brief description of the project structure:

* ansible: ansible playbook to automate the training steps on the remote machine.
* terraform: terraform files to provision the infrastructure.
* scripts: auxility shell scripts to automate repetitive tasks.
* infra_state: text files storing EC2 instance id and DNS name to be used in the scripts.

* datasets
    * funsd: the original funsd dataset
    * generated: the pre-processed annotations.
* bin: driver scripts for launching the training, data pre-processing, wrapper creation.
* config: yaml files with training hyper-parameters for local and remote training.
* model_artifacts: all the trained models, storing only DVC references, actual artifacts stored on S3.
* models: python code for the model architecture, dataset, vectorizer and vocabulary.
* preprocessing: basic python pre-processing logic to generate single-line JSON annotations.
* serving: python code for the model wrapper.
* synthetic: generated synthetic data, storing only DVC references, actual artifacts stored on S3.
* tools: python code to invoke the model locally and remotely.
* training: python code with the training loop.
* utilities: python code for file and image utilities.
