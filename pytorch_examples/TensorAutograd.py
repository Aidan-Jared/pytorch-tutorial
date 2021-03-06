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

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        y_pred = x.mm(w1).clamp(min = 0).mm(w2)

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss)
        
        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()