import torch
import mlflow

from models.annotations_dataset import AnnotationsDataset

class LSTMAnnotationsWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.lstm_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        self.dataset = AnnotationsDataset.load(context.artifacts["dataset_input_file"])
        self.vectorizer = self.dataset.get_vectorizer()
        self.vocab = self.vectorizer.get_vocabulary()

    def predict(self, context, model_input):
        print(f"Getting model inputs {model_input} type {type(model_input)}")
        model, vectorizer, vocab = self.lstm_model, self.vectorizer, self.vocab
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict_len=100
        temperature=0.8

        prime_str = vocab.START_SEQ
        hidden = model.init_zero_state(device)
        cell = model.init_zero_state(device)
        prime_input = vectorizer.vectorize(prime_str, wrap=False)
        predicted = model_input

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
        
        print(f"finally predicted  {predicted}")
        return predicted
