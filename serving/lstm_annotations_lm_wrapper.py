import torch
import mlflow
import string
import random

from models.annotations_dataset import AnnotationsDataset
from models.lstm_model_sampling import sample_from_model, decode_samples, sample_with_prompt, sample_from_model_with_prompt


class LSTMAnnotationsWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        if torch.cuda.is_available():
            self.lstm_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        else:
            kwargs = {"map_location":torch.device('cpu')}
            self.lstm_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"], **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model.eval()
        self.dataset = AnnotationsDataset.load(context.artifacts["dataset_input_file"])
        self.vectorizer = self.dataset.get_vectorizer()
        self.vocab = self.vectorizer.get_vocabulary()

    def predict(self, context, model_input):
        pass

    def predict3(self, context, model_input):
        print(f"Getting model inputs {model_input} type {type(model_input)}")
        model, vectorizer, vocab = self.lstm_model, self.vectorizer, self.vocab

        row = model_input.iloc[0]

        num_samples, sample_size, temperature = int(row.num_samples), int(row.sample_size), row.temperature

        samples = sample_from_model(model, vectorizer, self.device, num_samples, sample_size, temperature)
        
        sampled_annotations = decode_samples(samples, vectorizer)
        
        print(f"sampled_annotations {sampled_annotations}")

        return sampled_annotations

    def predict2(self, context, model_input):
        print(f"Getting model inputs {model_input} type {type(model_input)}")
        model, vectorizer, vocab = self.lstm_model, self.vectorizer, self.vocab

        row = model_input.iloc[0]

        num_samples, sample_size, temperature = int(row.num_samples), int(row.sample_size), row.temperature

        prompt_str = sample_text = "{" + f"\"text\": \"{string.printable[random.randint(10, 61)]}" 
        
        samples = sample_from_model_with_prompt(model, vectorizer, 
                                                self.device, prompt_str, 
                                                num_samples=2, sample_size=100)

        sampled_annotations = decode_samples(samples, vectorizer)

        # samples = sample_from_model(model, vectorizer, self.device, num_samples, sample_size, temperature)
        
        # sampled_annotations = decode_samples(samples, vectorizer)
        
        print(f"sampled_annotations {sampled_annotations}")

        return sampled_annotations

        