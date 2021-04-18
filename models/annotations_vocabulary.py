class AnnotationsVocabulary():
    START_SEQ = "<begin>"
    END_SEQ = "<end>"

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<unk>", add_seq_wrap=True):
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._add_seq_wrap = add_seq_wrap

        if not token_to_idx:
            self._token_to_idx = {}
            self._idx_to_token = {}
            
            if self._add_unk:
                self._unk_index = self.add_token(self._unk_token)

            if self._add_seq_wrap:
                self.begin_seq_index = self.add_token(AnnotationsVocabulary.START_SEQ)
                self.end_seq_index = self.add_token(AnnotationsVocabulary.END_SEQ)
        else:
            self._idx_to_token = {idx:token for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {'_token_to_idx': self._token_to_idx,
                '_add_unk': self._add_unk,
                '_unk_token': self._unk_token,
                '_add_seq_wrap': self._add_seq_wrap}

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
        if self._add_unk:
            return self._token_to_idx.get(token, self._unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, idx):
        if idx not in self._idx_to_token:
            raise KeyError(f"Unable to lookup token for index {idx} in Vocabulary")
        return self._idx_to_token[idx]

    def __len__(self):
        return len(self._token_to_idx)