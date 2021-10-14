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
    grad = -tx.T.dot(e) / len(e)
    return grad, e


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):

        # calculate the gradient
        grad, e = compute_gradient(y, tx, w)

        # update w
        w = w - gamma * grad

    return w, compute_loss(y, tx, w)


def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):

            # compute a stochastic gradient and loss
            grad, e = compute_stoch_gradient(y_batch, tx_batch, w)

            # update w
            w = w - gamma * grad

    return w, compute_loss(y, tx, w)


def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    lambda_derived = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_derived
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_loss(y, tx, w)
