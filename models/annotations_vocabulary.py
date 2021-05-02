class AnnotationsVocabulary():
    START_SEQ = "<begin>"
    END_SEQ = "<end>"

    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        if not token_to_idx:
            self._token_to_idx = {}
        
        self._idx_to_token = {}

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        return {'_token_to_idx': self._token_to_idx,
                'unk_token': self._unk_token,
                'mask_token':self._mask_token, 
                'begin_seq_token':self._begin_seq_token,
                'end_seq_token':self._end_seq_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            idx = self._token_to_idx[token]
        else:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
        return idx

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, idx):
        if idx not in self._idx_to_token:
            raise KeyError(f"Unable to lookup token for index {idx} in Vocabulary")
        return self._idx_to_token[idx]

    def __len__(self):
        return len(self._token_to_idx)