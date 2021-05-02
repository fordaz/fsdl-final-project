import time
from argparse import Namespace
import random
import cloudpickle
from sys import version_info

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
from models.lstm_annotations_lm_wrapper import LSTMAnnotationsWrapper
from training.training_context import TrainingContext

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


def generate_batches(dataset, batch_size, shuffle=False,
                     drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def evaluate(model, device, vectorizer, vocab, predict_len=100, temperature=0.8):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    prime_str = vocab.START_SEQ
    hidden = model.init_zero_state(device)
    cell = model.init_zero_state(device)
    prime_input = vectorizer.vectorize(prime_str, wrap=False)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = model(prime_input[p].to(device), (hidden.to(device), cell.to(device)))
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, (hidden, cell) = model(inp.to(device), (hidden.to(device), cell.to(device)))
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = vocab.lookup_index(top_i.item())
        if predicted_char == vocab.get_unk_token():
            continue
        if predicted_char == vocab.END_SEQ or predicted_char == vocab.START_SEQ:
            break
        predicted += predicted_char
        inp = vectorizer.vectorize(predicted_char, wrap=False)

    return predicted


def save_model(model, device, vectorizer, saved_model_fname, dataset_fname):
    mlflow.pytorch.save_model(pytorch_model=model, path=saved_model_fname)

    artifacts = {
        "pytorch_model": saved_model_fname,
        "dataset_input_file": dataset_fname
    }

    conda_env = {
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            "pytorch=1.8.1",
            "torchvision=0.9.1",
            'pip',
            {
                'pip': [
                    'mlflow',
                    'cloudpickle=={}'.format(cloudpickle.__version__),
                ],
            },
        ],
        'name': 'mlflow-env-wrapper'
    }

    mlflow_pyfunc_model_fname = f"{saved_model_fname}_wrapper"
    mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_fname, code_path=["training", "models"],
                             python_model=LSTMAnnotationsWrapper(), 
                             artifacts=artifacts, conda_env=conda_env)


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def sequence_loss(y_pred, y_true, mask_index):
    # print(f"y_pred {type(y_pred)}, y_true {type(y_true)}")
    # print(f"1. y_pred {y_pred.shape}, y_true {y_true.shape}")
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    # print(f"2. y_pred {y_pred.shape}, y_true {y_true.shape}")
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sample_from_model(model, vectorizer, num_samples=1, sample_size=20, temperature=1.0):
    vocab = vectorizer.get_vocabulary()
    begin_seq_index = [vocab.begin_seq_index 
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        probability_vector = model.sample(x_t, h_t, temperature)
        picked_indices = torch.multinomial(probability_vector, num_samples=1)
        indices.append(picked_indices)
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices


def decode_samples(sampled_indices, vectorizer):
    decoded_annotations = []
    vocab = vectorizer.get_vocabulary()
    
    for sample_index in range(sampled_indices.shape[0]):
        generated_annotation = ""
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index, time_step].item()
            if sample_item == vocab.begin_seq_index or sample_item == vocab.unk_index:
                continue
            elif sample_item == vocab.end_seq_index:
                break
            else:
                generated_annotation += vocab.lookup_index(sample_item)
        decoded_annotations.append(generated_annotation)
    return decoded_annotations


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

        # --------------------------------------    
        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred, _ = model(batch_dict['x_data'], (hidden, cell))

        # step 3. compute the loss
        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

        # step 4. use loss to produce gradients
        loss.backward()

        # step 5. use optimizer to take gradient step
        optimizer.step()

        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        if batch_index % args.batch_check_point == 0:
            print(f"Train: Have processed {batch_index} running_loss {running_loss}, running_acc {running_acc}")
    
    return {"train_loss": running_loss, "train_acc": running_acc}

def eval_on_batches(dataset, model, mask_index, args):
    dataset.set_split('val')
    batch_generator = generate_batches(dataset, 
                                        batch_size=args.batch_size, 
                                        device=args.device)
    running_loss, running_acc = 0.0, 0.0

    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        hidden = model.init_zero_state(args.device, args.batch_size)
        cell = model.init_zero_state(args.device, args.batch_size)

        # compute the output
        y_pred, _ = model(batch_dict['x_data'], (hidden, cell))

        # step 3. compute the loss
        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

        # compute the  running loss and running accuracy
        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        if batch_index % args.batch_check_point == 0:
            print(f"Val: Have processed {batch_index} running_loss {running_loss}, running_acc {running_acc}")
    
    return {"val_loss": running_loss, "val_acc": running_acc}


def train(dataset, saved_model_fname, dataset_fname, args):
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

    mlflow.start_run()
    mlflow.log_params(vars(args))

    train_ctx = TrainingContext(args)

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        train_metrics = train_on_batches(dataset, model, optimizer, mask_index, args)
        
        train_ctx.append_metrics(train_metrics)
        mlflow.log_metrics(train_metrics)

        val_metrics = eval_on_batches(dataset, model, mask_index, args)

        train_ctx.append_metrics(val_metrics)
        mlflow.log_metrics(val_metrics)

        train_ctx.update(model, epoch, saved_model_fname)

        samples = sample_from_model(model, vectorizer, num_samples=2)
        sampled_annotations = decode_samples(samples, vectorizer)
        print(f"sampled_annotations {sampled_annotations}")

        if train_ctx.stop_early:
            print(f"Early stopping, best validation loss {train_ctx.early_stopping_best_val}")
            break
    
    mlflow.pytorch.save_model(pytorch_model=model, path=saved_model_fname)

def train_driver(dataset_fname, saved_model_fname, args):
    dataset = AnnotationsDataset.load(dataset_fname)
    train(dataset, saved_model_fname, dataset_fname, args)
