from models.annotations_vectorizer import AnnotationsVectorizer

from torch.utils.data import Dataset
import pandas as pd

class AnnotationsDataset(Dataset):
    def __init__(self, annotations, vectorizer):
        self._annotations = annotations
        self._vectorizer = vectorizer
        self._annotations_len = len(self._annotations)

    @classmethod
    def load(cls, input_fname):
        with open(input_fname, 'r') as f:
            annotations = f.readlines()
        return cls(annotations, AnnotationsVectorizer.from_text(annotations))

    def __len__(self):
        return self._annotations_len

    def __getitem__(self, index):
        annotation = self._annotations[index]
        vect_annotation = self._vectorizer.vectorize(annotation)
        return vect_annotation

    def get_num_batches(self, batch_size):
        return self._annotations_len // batch_size

    def get_vectorizer(self):
        return self._vectorizer