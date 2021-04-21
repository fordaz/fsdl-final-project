import torch
from models.annotations_vocabulary import AnnotationsVocabulary
import string

class AnnotationsVectorizer():
    def __init__(self, annotations_vocab):
        self._vocab = annotations_vocab

    def vectorize(self, annotation, wrap=True):
        lst = []

        if wrap:
            lst.extend([self._vocab.begin_seq_index])

        lst.extend([self._vocab.lookup_token(character) for character in annotation])

        if wrap:
            lst.extend([self._vocab.end_seq_index])

        return torch.tensor(lst).long()

        # annotation_tensor = torch.tensor(lst).long()

        # inputs = annotation_tensor[:-1]
        # targets = annotation_tensor[1:]

        # return inputs, targets

    def vectorize_char(self, character):
        return torch.tensor([self._vocab.lookup_token(character)]).long()

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
    def from_serializable(cls, contents):
        vocab = AnnotationsVocabulary.from_serializable(contents['vocab'])
        return cls(vocab)

    def to_serializable(self):
        return {'vocab': self._vocab}
    
    def get_vocabulary(self):
        return self._vocab