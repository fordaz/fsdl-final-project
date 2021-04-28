import torch
import mlflow

from models.annotations_dataset import AnnotationsDataset

class LSTMAnnotationsWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        if torch.cuda.is_available():
            self.lstm_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        else:
            kwargs = {"map_location":torch.device('cpu')}
            self.lstm_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"], **kwargs)
        self.lstm_model.eval()
        self.dataset = AnnotationsDataset.load(context.artifacts["dataset_input_file"])
        self.vectorizer = self.dataset.get_vectorizer()
        self.vocab = self.vectorizer.get_vocabulary()

    def predict(self, context, model_input):
        print(f"Getting model inputs {model_input} type {type(model_input)}")
        model, vectorizer, vocab = self.lstm_model, self.vectorizer, self.vocab
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict_len=500
        temperature=0.6

        predicted = model_input
        hidden = model.init_zero_state(device)
        cell = model.init_zero_state(device)
        prime_input = vectorizer.vectorize(model_input, wrap=False)

        # Use priming string to "build up" hidden state
        for p in range(len(prime_input) - 1):
            _, (hidden, cell) = model(prime_input[p].to(device), (hidden.to(device), cell.to(device)))
        inp = prime_input[-1]
        # inp = model_input
        
        for p in range(predict_len):
            # print(f"evaluate inputs inp {inp.shape} hidden {hidden.shape} cell {cell.shape}")
            output, (hidden, cell) = model(inp.to(device), (hidden.to(device), cell.to(device)))
            
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input
            predicted_char = vocab.lookup_index(top_i.item())
            # print(f"predicted_char {predicted_char}")
            if predicted_char == vocab.get_unk_token():
                continue
            if predicted_char == vocab.END_SEQ or predicted_char == vocab.START_SEQ:
                break
            predicted += predicted_char
            inp = vectorizer.vectorize(predicted_char, wrap=False)
        
        # print(f"finally predicted  {predicted}")
        return predicted
