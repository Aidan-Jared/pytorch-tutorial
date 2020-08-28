import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size, layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.category_size = n_categories
        self.layers = layers

        self.input_2_output = nn.LSTM(n_categories + input_size, hidden_size, self.layers)
        self.output_2_output = nn.Linear(n_categories + hidden_size, output_size)
        self.dropout = nn.Dropout(.1) # prevent over fitting / increase sampling variety
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden, cell_state):
        input = input.view(1,1,self.input_size)
        category = category.view(1,1,self.category_size)
        input_combined = torch.cat((category, input), 2)
        
        output, (hidden, cell_state) = self.input_2_output(input_combined, (hidden, cell_state))       
        # output = F.relu(output)
        output_combined = torch.cat((category, output), 2)
        
        output = self.output_2_output(output_combined)
        
        output = self.dropout(output)
        output = output.view(1,-1)
        output = self.softmax(output)
        return output, hidden, cell_state

    def initHidden(self):
        return torch.zeros(self.layers, 1, self.hidden_size)