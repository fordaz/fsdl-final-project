import torch
from models.annotations_vocabulary import AnnotationsVocabulary
import string

class AnnotationsVectorizer():
    def __init__(self, annotations_vocab):
        self._vocab = annotations_vocab

    def vectorize(self, annotation):
        lst = [self._vocab.begin_seq_index]
        lst.extend([self._vocab.lookup_token(character) for character in annotation])
        lst.extend([self._vocab.end_seq_index])

        tensor = torch.tensor(lst).long()
        return tensor

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