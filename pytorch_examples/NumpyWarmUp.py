import numpy as np

if __name__ == "__main__":
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = 64
    D_in = 1000
    H = 100
    D_out = 10

    x = np.random.rand(N,D_in)
    y = np.random.randn(N,D_out)

    # initialize weights
    w1 = np.random.rand(D_in, H)
    w2 = np.random.rand(H, D_out)

    learning_rate = 1

    for t in range(500):
        # foward pass
        h = x.dot(w1)
        h_relu = np.maximum(h,0)
        y_pred = h_relu.dot(w2)

        loss = np.square(y_pred - y).sum()
        print(t, loss)

        grad_y_pred = 2 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        # grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(loss)