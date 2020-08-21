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

    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        h = x.mm(w1)
        h_relu = h.clamp(min = 0)
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h<0] = 0
        grad_w1 = x.t().mm(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2