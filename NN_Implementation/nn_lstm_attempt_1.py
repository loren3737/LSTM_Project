import torch

class LSTM(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout,
                 bidirectional, num_embeddings, enforce_sorted=True):
        super().__init__()

        self.bidirectional = bidirectional
        self.enforceSorted = enforce_sorted

        # embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings, num_features)

        # LSTM layer
        self.lstm = torch.nn.LSTM(num_features, hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=dropout, bidirectional=bidirectional)

        # output layer with single output
        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1),
                                               torch.nn.Sigmoid())

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # pack padding so that all sequences (reviews) are the same length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=self.enforceSorted)
        
        _, (hidden_state, _) = self.lstm(packed)

        if self.bidirectional:
            # concat the forward and backward hidden states of the last layer
            out = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        else:
            # use the hidden state of the last layer
            out = hidden_state[-1, :, :]
        
        out = self.outputLayer(out)

        return out
