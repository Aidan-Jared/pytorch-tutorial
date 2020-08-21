import torch

if __name__ == "__main__":
    dtype = torch.float
    device = torch.device('cpu')

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = 64
    D_in = 1000
    H = 100
    D_out = 10

    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out)
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(500):
        y_pred = model(x)

        loss = loss_fn(y_pred,y)

        if t % 100 == 99:
            print(t, loss.item())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
