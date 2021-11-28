import torch
from torch import nn


class RNNPred(nn.Module):
    def __init__(self, input_dim: int, seq_length: int, hidden_size: int, no_of_hidden_layers: int, output_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.no_of_hidden_layers = no_of_hidden_layers
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=no_of_hidden_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * seq_length, output_size)

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        # hidden_state = self.init_hidden(16)
        # reshaped_input = input.view(1, 1, 10)
        output, _ = self.rnn(input, self.hidden_state)
        output = output.contiguous().view(batch_size, -1)
        output = self.linear(output)
        return output

    def init_hidden(self, batch_size=16):
        self.hidden_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()





class MV_LSTM(torch.nn.Module):
    """
    ONLY FOR REFERENCE
    """
    # https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch

    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 128  # number of hidden states
        self.n_layers = 3  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
