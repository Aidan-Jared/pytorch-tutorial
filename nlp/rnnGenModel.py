import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_2_hidden = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.input_2_output = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.output_2_output = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(.1) # prevent over fitting / increase sampling variety
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.input_2_hidden(input_combined)
        output = self.input_2_output(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.output_2_output(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)