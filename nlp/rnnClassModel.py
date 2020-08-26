import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.input_2_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_2_output = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # combined = torch.cat((input, hidden), 1)
        # hidden = self.input_2_hidden(combined)

        #test input should be in shape of (batch size, time steps, seq length)
        input.unsqueeze_(-1)
        input = input.expand(1,1,self.input_size)
        output, hidden = self.input_2_output(input, hidden)
        output = self.out(output[:,1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.input_size, self.hidden_size)