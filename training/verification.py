import mlflow
import torch
from models.annotations_dataset import AnnotationsDataset
from models.annotations_vectorizer import AnnotationsVectorizer

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


if __name__ == "__main__":
    # save_model_dir = "saved_model_001"
    # dataset_fname = "./training/test_input_annotations.txt"
    # verify(dataset_fname, save_model_dir)
    
    save_model_dir = "saved_model_001_wrapper"
    dataset_fname = "./training/test_input_annotations.txt"
    verify2(dataset_fname, save_model_dir)
    