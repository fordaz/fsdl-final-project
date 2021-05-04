import torch
import mlflow

from models.annotations_dataset import AnnotationsDataset
from models.lstm_model_sampling import sample_from_model, decode_samples


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

        row = model_input.iloc[0]

        num_samples, sample_size, temperature = int(row.num_samples), int(row.sample_size), row.temperature

        samples = sample_from_model(model, vectorizer, num_samples, sample_size, temperature)
        
        sampled_annotations = decode_samples(samples, vectorizer)
        
        print(f"sampled_annotations {sampled_annotations}")

        return sampled_annotations