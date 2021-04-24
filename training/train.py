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


RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

args = Namespace(
    embedding_dim=100,
    hidden_dim=100,
    num_hidden=1,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    learning_rate=0.005,
    epochs=3,
    epoch_check_point=1,
    sample_text_len=500
)


def sample_dataset(dataset, device):
    idx = random.randint(0, len(dataset)-1)
    inputs = dataset[idx]
    inputs = torch.squeeze(inputs)
    inputs, targets = inputs[:-1], inputs[1:]
    return inputs.to(device), targets.to(device)


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


def train(dataset, saved_model_fname, dataset_fname):
    vectorizer = dataset.get_vectorizer()
    vocab = vectorizer.get_vocabulary()
    vocab_len = len(vocab)
    print(f"Training using device {args.device}")

    model = LSTMAnnotationsLM(vocab_len, args.embedding_dim, args.hidden_dim, vocab_len, args.num_hidden)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_time = time.time()
    try:
        mlflow.start_run()
        mlflow.log_param("epochs", str(args.epochs))

        for epoch in range(args.epochs):
            hidden = model.init_zero_state(args.device)
            cell = model.init_zero_state(args.device)

            optimizer.zero_grad()
            loss = 0.
            inputs, targets = sample_dataset(dataset, args.device)
            input_length = len(inputs)
            for c in range(input_length):
                outputs, (hidden, cell) = model(inputs[c], (hidden, cell))
                loss += F.cross_entropy(outputs, targets[c].view(1))
            loss /= input_length
            loss.backward()
            optimizer.step()

            with torch.set_grad_enabled(False):
                if epoch % args.epoch_check_point == 0:
                    mlflow.log_metrics({"epoch": epoch, "loss": loss.item()})
                    mlflow.pytorch.log_model(model, saved_model_fname)
                    # torch.save(model, saved_model_fname)
                    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
                    print(f'Epoch {epoch} | Loss {loss.item():.2f}\n\n')
                    print(evaluate(model, args.device, vectorizer, vocab, args.sample_text_len), '\n')
                    print(50*'=')
        save_model(model, args.device, vectorizer, saved_model_fname, dataset_fname)

    finally:
        mlflow.end_run()


def train_driver(dataset_fname, saved_model_fname):
    dataset = AnnotationsDataset.load(dataset_fname)
    train(dataset, saved_model_fname, dataset_fname)

