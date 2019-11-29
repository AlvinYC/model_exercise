import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class MORVANLSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
    
class AlvinLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(AlvinLSTM, self).__init__()    

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=hidden_dim, # rnn hidden unit
            num_layers=layer_dim,   # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out   
    
class AlvinGRU(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(AlvinGRU, self).__init__()    

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=hidden_dim, # rnn hidden unit
            num_layers=layer_dim,   # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        
        r_out, hn = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out       