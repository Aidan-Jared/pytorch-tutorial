import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.category_size = n_categories

        # self.input_2_hidden = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.input_2_output = nn.GRU(n_categories + input_size, hidden_size)
        self.output_2_output = nn.Linear(n_categories + hidden_size, output_size)
        self.dropout = nn.Dropout(.1) # prevent over fitting / increase sampling variety
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input = input.view(1,1,self.input_size)
        category = category.view(1,1,self.category_size)
        input_combined = torch.cat((category, input), 2)
        # hidden = self.input_2_hidden(input_combined)
        output, hidden = self.input_2_output(input_combined, hidden)
        output_combined = torch.cat((category, output), 2)
        output = self.output_2_output(output_combined)
        output = self.dropout(output)
        output = output.view(1,-1)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)