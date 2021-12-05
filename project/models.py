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
        output, self.hidden = self.rnn(input, self.hidden)
        output = output.contiguous().view(batch_size, -1)
        output = self.linear(output)
        return output

    def init_hidden(self, batch_size=16):
        hidden_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()
        # cell_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()
        # self.hidden = (hidden_state, cell_state)
        self.hidden = hidden_state


class GRUPred(nn.Module):
    def __init__(self, input_dim: int, seq_length: int, hidden_size: int, no_of_hidden_layers: int, output_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.no_of_hidden_layers = no_of_hidden_layers
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers=no_of_hidden_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * seq_length, output_size)

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        output, self.hidden = self.rnn(input, self.hidden)
        output = output.contiguous().view(batch_size, -1)
        output = self.linear(output)
        return output

    def init_hidden(self, batch_size=16):
        hidden_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()
        self.hidden = hidden_state


class LSTMPred(nn.Module):
    # Ref: https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
    def __init__(self, input_dim: int, seq_length: int, hidden_size: int, no_of_hidden_layers: int, output_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.no_of_hidden_layers = no_of_hidden_layers
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=no_of_hidden_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * seq_length, output_size)

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        # hidden_state = self.init_hidden(16)
        # reshaped_input = input.view(1, 1, 10)
        output, self.hidden = self.rnn(input, self.hidden)
        output = output.contiguous().view(batch_size, -1)
        output = self.linear(output)
        return output

    def init_hidden(self, batch_size=16):
        hidden_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()
        cell_state = torch.zeros(self.no_of_hidden_layers, batch_size, self.hidden_size).requires_grad_()
        self.hidden = (hidden_state, cell_state)
