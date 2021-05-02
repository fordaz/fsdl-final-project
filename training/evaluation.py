import string
import random
import mlflow
import torch
import pandas as pd

from models.annotations_dataset import AnnotationsDataset
from models.annotations_vectorizer import AnnotationsVectorizer
from models.lstm_annotations_lm_wrapper import LSTMAnnotationsWrapper
from collections import namedtuple

def evaluate(model, device, vectorizer, vocab, predict_len=100, temperature=0.8):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    prime_str = vocab.START_SEQ
    hidden = model.init_zero_state(device)
    cell = model.init_zero_state(device)
    prime_input = vectorizer.vectorize(prime_str, wrap=False)
    predicted = prime_str

    print(type(prime_input), prime_input.shape, prime_input)
    print(type(hidden), hidden.shape, hidden)
    print(type(cell), cell.shape, cell)

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = model(prime_input[p].to(device), (hidden.to(device), cell.to(device)))
    inp = prime_input[-1]
    
    for p in range(predict_len):
        print(f"evaluate inputs inp {inp.shape} hidden {hidden.shape} cell {cell.shape}")
        output, (hidden, cell) = model(inp.to(device), (hidden.to(device), cell.to(device)))
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = vocab.lookup_index(top_i.item())
        print(f"predicted_char {predicted_char}")
        if predicted_char == vocab.get_unk_token():
            continue
        if predicted_char == vocab.END_SEQ or predicted_char == vocab.START_SEQ:
            break
        predicted += predicted_char
        inp = vectorizer.vectorize(predicted_char, wrap=False)

    return predicted


def evaluate2(model, dataset_fname):
    vectorizer = AnnotationsVectorizer.from_text(dataset_fname)
    vocab = vectorizer.get_vocabulary()
    prime_str = "{"
    # vocab.START_SEQ
    prime_input = vectorizer.vectorize(prime_str, wrap=False)
    return model.predict(prime_input)


def verify(dataset_fname, saved_model_fname):
    model = mlflow.pytorch.load_model(saved_model_fname)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AnnotationsDataset.load(dataset_fname)
    vectorizer = dataset.get_vectorizer()
    vocab = vectorizer.get_vocabulary()
    print(evaluate(model, device, vectorizer, vocab))


def verify2(dataset_fname, saved_model_fname):
    model = mlflow.pyfunc.load_model(saved_model_fname)
    print(f"verified {evaluate2(model, dataset_fname)}")


def verify3(dataset_fname, saved_model_fname):
    device = torch.device('cpu')
    kwargs = {"map_location":device}
    ml_model = mlflow.pytorch.load_model(model_uri=saved_model_fname, **kwargs)
    artifacts = {
        "pytorch_model": saved_model_fname,
        "dataset_input_file": dataset_fname
    }
    Context = namedtuple("Context", ["artifacts"])
    
    lstm_wrapper = LSTMAnnotationsWrapper()
    ctx = Context(artifacts)
    lstm_wrapper.load_context(ctx)
    
    annotations_df = pd.read_csv(dataset_fname)
    vectorizer = AnnotationsVectorizer.from_dataframe(annotations_df)
    vocab = vectorizer.get_vocabulary()

    model_input_param = {"num_samples": 3, "sample_size": 50, "temperature": 1.0}

    print(lstm_wrapper.predict(ctx, model_input_param))


if __name__ == "__main__":
    save_model_dir = "model_artifacts/saved_model_local"
    dataset_fname = "datasets/generated/full/annotations/full_annotations_small.csv"
    # verify2(dataset_fname, save_model_dir)
    verify3(dataset_fname, save_model_dir)
    