import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet,self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min = 0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min = 0)
        y_pred = self.output_linear(h_relu)
        return y_pred

if __name__ == "__main__":
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = 64
    D_in = 1000
    H = 100
    D_out = 10

    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)

    model = DynamicNet(D_in, H, D_out)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum=.9)

    for t in range(500):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()