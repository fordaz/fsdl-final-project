import torch
import torch.nn.functional as F

class LSTMAnnotationsLM(torch.nn.Module):
    """
    This is an adaptation of the source code of the book (Chapter 7): 
    Natural Language Processing with PyTorch, by Delip Rao and Brian McMahan
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, padding_idx, batch_first=True, dropout_p=0.5):
        super(LSTMAnnotationsLM, self).__init__()

        self.num_layers = num_layers
        self.num_directions = 2
        self.hidden_size = hidden_size
        
        self.embed = torch.nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_size,
                                        padding_idx=padding_idx)

        is_bidirectional = True if self.num_directions == 2 else False
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, bidirectional=is_bidirectional, 
                                  batch_first=batch_first)

        self.fc = torch.nn.Linear(hidden_size*self.num_directions, vocab_size)

        self._dropout_p = dropout_p

    def forward(self, features, hidden_and_cell):
        # print(f"1. Forward {features.shape}")
        embeddings = self.embed(features)

        # print(f"2. Forward {embeddings.shape} {hidden_and_cell[0].shape} {hidden_and_cell[1].shape}")
        output, (hidden, cell) = self.lstm(embeddings, hidden_and_cell)

        batch_size, seq_size, feat_size = output.shape
        output = output.contiguous().view(batch_size*seq_size, feat_size)

        output = self.fc(F.dropout(output, p=self._dropout_p))
                         
        new_feat_size = output.shape[-1]
        output = output.view(batch_size, seq_size, new_feat_size)

        # print(f"3. Forward {output.shape} {output.view(1, -1).shape} {self.fc.weight.shape} {self.fc.bias.shape}")
        return output, (hidden, cell)
      
    def init_zero_state(self, device, batch_size):
        return torch.zeros(self.num_layers*self.num_directions, batch_size, 
                           self.hidden_size).to(device)

    def sample(self, features, hidden_and_cell, temperature):
        embeddings = self.embed(features)
        rnn_out_t, h_t = self.lstm(embeddings, hidden_and_cell)
        prediction_vector = self.fc(rnn_out_t.squeeze(dim=1))
        return F.softmax(prediction_vector / temperature, dim=1), h_t
