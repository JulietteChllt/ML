import numpy as np


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


# given function
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


# ---- logistic regression ----

def log_of_sigmoid(x):
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


#Optimized negative log-likelihood function (using log_of_sigmoid optimisations)
def calculate_loss(x, A, b):

    z = np.dot(tx, w)
    y = np.asarray(y)
    return np.mean((1 - y) * z - log_of_sigmoid(z))

#Logistic sigmoid function (inverse of the logit function)
def expit_y(y_pred, y):
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


#Optimized Gradient Descent for logistic regression (using expit function)
def calculate_gradient(w, tx, y):
    y_pred = tx.dot(w)
    s = expit_y(y_pred, y)
    return tx.T.dot(s) / tx.shape[0] 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iters):
        loss = calculate_loss(w, tx, y)
        gradient  = calculate_gradient(w, tx, y)
        print(loss)
        w= w - gamma*gradient

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("converged after this amount of iterations:",n_iter)
            break
            
    return w.squeeze(),loss


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):    
    threshold = 1e-4
    losses = []
    w = initial_w
  
    for iter in range(max_iters):
        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w= w - gamma*gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w.squeeze(),loss
