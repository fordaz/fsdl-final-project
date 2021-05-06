import torch
from models.annotations_vocabulary import AnnotationsVocabulary
import string
import numpy as np

class AnnotationsVectorizer():
    """
    This is an adaptation of the source code of the book (Chapter 7): 
    Natural Language Processing with PyTorch, by Delip Rao and Brian McMahan
    """
    def __init__(self, annotations_vocab):
        self._vocab = annotations_vocab

    def vectorize(self, annotation, vector_length=-1):
        indices = [self._vocab.begin_seq_index]
        indices.extend([self._vocab.lookup_token(character) for character in annotation])
        indices.extend([self._vocab.end_seq_index])

        if vector_length < 0:
            vector_length = len(indices) - 1

        from_vector = np.empty(vector_length, dtype=np.int64)
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self._vocab.mask_index

        to_vector = np.empty(vector_length, dtype=np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self._vocab.mask_index
        
        return from_vector, to_vector

    def vectorize_char(self, character):
        return torch.tensor([self._vocab.lookup_token(character)])

    @classmethod
    def from_text(cls, annotations):
        vocab = AnnotationsVocabulary()

        for token in string.printable:
            vocab.add_token(token)

        for annotation in annotations:
            for token in annotation:
                vocab.add_token(token)
        
        return cls(vocab)
    
    @classmethod
    def from_dataframe(cls, annotations_df):
        vocab = AnnotationsVocabulary()

        for token in string.printable:
            vocab.add_token(token)

        for index, row in annotations_df.iterrows():
            for token in row.annotation:
                vocab.add_token(token)
        
        return cls(vocab)

    @classmethod
    def from_serializable(cls, contents):
        vocab = AnnotationsVocabulary.from_serializable(contents['vocab'])
        return cls(vocab)

    def to_serializable(self):
        return {'vocab': self._vocab}
    
    def get_vocabulary(self):
        return self._vocab