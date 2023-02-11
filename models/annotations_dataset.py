from models.annotations_vectorizer import AnnotationsVectorizer

from torch.utils.data import Dataset
import pandas as pd


class AnnotationsDataset(Dataset):
    """
    This is an adaptation of the source code of the book (Chapter 7):
    Natural Language Processing with PyTorch, by Delip Rao and Brian McMahan
    """

    def __init__(self, annotations_df, vectorizer):
        self.annotations_df = annotations_df
        self._vectorizer = vectorizer

        self._max_seq_length = max(map(len, self.annotations_df.annotation)) + 2

        self.train_df = self.annotations_df[self.annotations_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.annotations_df[self.annotations_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.annotations_df[self.annotations_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    @classmethod
    def load(cls, input_fname):
        annotations_df = pd.read_csv(input_fname)
        return cls(annotations_df, AnnotationsVectorizer.from_dataframe(annotations_df))

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        from_vector, to_vector = self._vectorizer.vectorize(
            row.annotation, self._max_seq_length
        )
        return {"x_data": from_vector, "y_target": to_vector}

    def get_num_batches(self, batch_size):
        return self._target_size // batch_size

    def get_vectorizer(self):
        return self._vectorizer
