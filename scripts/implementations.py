import numpy as np


###### Proposition Augustin ########

def MSE(e):
    return np.mean(e**2)/2


def MAE(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, mse=True):
    e = y - tx.dot(w)
    if(mse):
        return MSE(e)
    else:
        return MAE(e)


def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/len(e)
    return grad, e


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):

        # calculate the gradient
        grad, e = compute_gradient(y, tx, w)

        # update the loss
        loss = MSE(e)

        # update w
        w = w - gamma * grad

    return loss, w
