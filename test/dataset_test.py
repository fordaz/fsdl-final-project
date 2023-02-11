import unittest

from models.annotations_dataset import AnnotationsDataset

class TestSum(unittest.TestCase):
    def test(self):
        input_fname = "test_input_annotations.txt"
        dataset = AnnotationsDataset.load(input_fname)

        self.assertFalse(dataset is None)
        self.assertEqual(len(dataset), 16)

        vec_annotation = dataset[0]
        self.assertEqual(vec_annotation[0].item(), dataset._vectorizer._vocab.begin_seq_index)
        self.assertEqual(vec_annotation[vec_annotation.shape[0]-1].item(), dataset._vectorizer._vocab.end_seq_index)

if __name__ == '__main__':
    unittest.main()
