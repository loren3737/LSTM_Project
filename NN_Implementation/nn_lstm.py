import torch

class LSTM(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout,
                 bidirectional):
        super().__init__()

        # LSTM layer
        self.lstm = torch.nn.LSTM(num_features, hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=dropout, bidirectional=bidirectional)

        # output layer with single output
        # TODO input num features may need to be changed if LSTM is
        # bidirectional
        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(hidden_size, 1),
                                               torch.nn.Sigmoid())

    def forward(self, input_batch):
        _, (hidden_state, _) = self.lstm(input_batch)

        # use the hidden state of the last layer
        out = hidden_state[-1, :, :]
        
        out = self.outputLayer(out)

        return out
