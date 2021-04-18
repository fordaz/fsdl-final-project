import unittest

from models.annotations_vocabulary import AnnotationsVocabulary

class TestSum(unittest.TestCase):
    def test(self):
        vocab = AnnotationsVocabulary()
        
        self.assertFalse(vocab is None)
        
        self.assertTrue(vocab.lookup_token(AnnotationsVocabulary.START_SEQ))
        self.assertTrue(vocab.lookup_token(AnnotationsVocabulary.END_SEQ))
        
        vocab.add_token("a")
        self.assertTrue(vocab.lookup_token("a"))
        
        self.assertEqual(len(vocab), 4)

if __name__ == '__main__':
    unittest.main()