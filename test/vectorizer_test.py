import unittest

from models.annotations_vectorizer import AnnotationsVectorizer


class TestSum(unittest.TestCase):
    def test(self):
        annotations = [
            '{"box": [84, 109, 136, 119], "text": "COMPOUND", "label": "question", "linking": [[0, 37]], "id": 0}',
            '{"box": [85, 141, 119, 152], "text": "SOURCE", "label": "question", "linking": [[1, 38]], "id": 1}',
        ]
        vectorizer = AnnotationsVectorizer.from_text(annotations)

        self.assertFalse(vectorizer is None)

        vec_annotation = vectorizer.vectorize(
            '{"box": [84, 109, 136, 119], "text": "COMPOUND", "label": "question", "linking": [[0, 37]], "id": 0}'
        )

        self.assertEqual(vec_annotation[0].item(), vectorizer._vocab.begin_seq_index)
        self.assertEqual(
            vec_annotation[vec_annotation.shape[0] - 1].item(),
            vectorizer._vocab.end_seq_index,
        )


if __name__ == "__main__":
    unittest.main()
