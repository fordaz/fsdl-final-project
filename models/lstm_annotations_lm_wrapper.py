import torch
import mlflow

from models.annotations_dataset import AnnotationsDataset

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
        
        num_samples = model_input['num_samples']
        sample_size = model_input['sample_size']
        temperature = model_input['temperature']

        samples = sample_from_model(model, vectorizer, num_samples, sample_size, temperature)
        sampled_annotations = decode_samples(samples, vectorizer)
        print(f"sampled_annotations {sampled_annotations}")

        return sampled_annotations