import torch

class LSTMAnnotationsLM(torch.nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers):
        super(LSTMAnnotationsLM, self).__init__()

        self.num_layers = num_layers
        self.num_directions = 2
        self.hidden_size = hidden_size
        
        self.embed = torch.nn.Embedding(input_size, hidden_size)

        is_bidirectional = True if self.num_directions == 2 else False
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, bidirectional=is_bidirectional)

        self.fc = torch.nn.Linear(hidden_size*self.num_directions, output_size)

        self.init_hidden = torch.nn.Parameter(torch.zeros(self.num_layers*self.num_directions, 1, hidden_size))
    
    def forward(self, features, hidden_and_cell):
        # print(f"1. Forward {features.shape}")
        embedded = self.embed(features.view(1, -1))
        # print(f"2. Forward {embedded.shape} {embedded.view(1, 1, -1).shape}")
        output, (hidden, cell) = self.lstm(embedded.view(1, 1, -1), hidden_and_cell)
        # print(f"3. Forward {output.shape} {output.view(1, -1).shape} {self.fc.weight.shape} {self.fc.bias.shape}")
        output = self.fc(output.view(1, -1))
        # print(f"4. Forward")
        return output, (hidden, cell)
      
    def init_zero_state(self, device):
        return torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size).to(device)
