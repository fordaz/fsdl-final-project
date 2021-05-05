import time
from argparse import Namespace
import random
import traceback

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import mlflow.pyfunc

from models.annotations_dataset import AnnotationsDataset
from models.lstm_annotations_lm import LSTMAnnotationsLM
from training.training_context import TrainingContext
from training.train_utils import (generate_batches, 
                                  normalize_sizes, 
                                  sequence_loss, 
                                  compute_accuracy, 
                                  set_seed)
from models.lstm_model_sampling import sample_model


def train_driver(dataset_fname, saved_model_fname, args):
    dataset = AnnotationsDataset.load(dataset_fname)
    train(dataset, saved_model_fname, dataset_fname, args)


def train(dataset, saved_model_fname, dataset_fname, args):
    set_seed(args.seed)

    vectorizer = dataset.get_vectorizer()
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    print(f"Training using device {args.device}")

    model = LSTMAnnotationsLM(vocab_size, args.embedding_dim, 
                              args.hidden_dim, args.num_layers, 
                              vocab.mask_index)

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    mask_index = vocab.mask_index

    try:
        mlflow.start_run()
        mlflow.log_params(vars(args))

        train_ctx = TrainingContext(args)

        for epoch in range(args.epochs):
            print(f"Starting epoch {epoch}")

            train_metrics = train_on_batches(dataset, model, optimizer, mask_index, args)
            
            train_ctx.append_metrics(train_metrics)
            mlflow.log_metrics(train_metrics)

            val_metrics = test_on_batches("val", dataset, model, mask_index, args)

            train_ctx.append_metrics(val_metrics)
            mlflow.log_metrics(val_metrics)

            train_ctx.update(model, epoch, saved_model_fname)

            sampled_annotations = sample_model(model, vectorizer)
            print(f"sampled_annotations {sampled_annotations}")

            if train_ctx.stop_early:
                print(f"Early stopping, best validation loss {train_ctx.early_stopping_best_val}")
                break

        test_metrics = test_on_batches("test", dataset, model, mask_index, args)

        train_ctx.append_metrics(test_metrics)
        mlflow.log_metrics(test_metrics)

        mlflow.pytorch.save_model(pytorch_model=model, path=saved_model_fname)
    except Exception as e:
        traceback.print_exception(e)
        print(f"Unexpected exception while training {e}")
    finally:
        mlflow.end_run()


def train_on_batches(dataset, model, optimizer, mask_index, args):
    dataset.set_split('train')
    batch_generator = generate_batches(dataset, 
                                        batch_size=args.batch_size, 
                                        device=args.device)

    running_loss, running_acc = 0.0, 0.0

    model.train()
    for batch_index, batch_dict in enumerate(batch_generator):
        hidden = model.init_zero_state(args.device, args.batch_size)
        cell = model.init_zero_state(args.device, args.batch_size)

        optimizer.zero_grad()

        y_pred, _ = model(batch_dict['x_data'], (hidden, cell))

        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

        loss.backward()

        optimizer.step()

        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        if batch_index % args.batch_check_point == 0:
            print(f"train: Have processed {batch_index} running_loss {running_loss}, running_acc {running_acc}")
    
    return {"train_loss": running_loss, "train_acc": running_acc}


def test_on_batches(test_type, dataset, model, mask_index, args):
    dataset.set_split(test_type)
    batch_generator = generate_batches(dataset, 
                                        batch_size=args.batch_size, 
                                        device=args.device)
    running_loss, running_acc = 0.0, 0.0

    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        hidden = model.init_zero_state(args.device, args.batch_size)
        cell = model.init_zero_state(args.device, args.batch_size)

        y_pred, _ = model(batch_dict['x_data'], (hidden, cell))

        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        if batch_index % args.batch_check_point == 0:
            print(f"{test_type}: Have processed {batch_index} running_loss {running_loss}, running_acc {running_acc}")
    
    return {f"{test_type}_loss": running_loss, f"{test_type}_acc": running_acc}




