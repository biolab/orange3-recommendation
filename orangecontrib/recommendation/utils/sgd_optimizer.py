import numpy as np


# Global variables
velocity = None
accu = None


def init_optimizers(num_grads):
    global velocity
    global accu

    velocity = [0] * num_grads
    accu = [0] * num_grads


def apply_momentum(updates, params, momentum):
    global velocity

    for i in range(len(params)):
        velocity[i] = momentum * velocity[i] + updates[i] - params[i]
        updates[i] = velocity[i]
    return updates


def apply_nesterov_momentum(updates, params, momentum):
    global velocity

    for i in range(len(params)):
        x = momentum * velocity[i] + updates[i] - params[i]
        velocity[i] = x
        updates[i] = momentum * x + updates[i]
    return updates


def sgd(grads, params, learning_rate):
    updates = []
    for i in range(len(params)):
        p = params[i] - learning_rate * grads[i]
        updates.append(p)
    return updates


def momentum(grads, params, learning_rate, momentum=0.9):
    updates = sgd(grads, params, learning_rate)
    return apply_momentum(updates, params, momentum=momentum)


def nesterov_momentum(grads, params, learning_rate, momentum=0.9):
    updates = sgd(grads, params, learning_rate)
    return apply_nesterov_momentum(updates, params, momentum=momentum)


def adagrad(grads, params, learning_rate=1.0, epsilon=1e-6):
    global velocity
    global accu

    # i = np.arange(len(params))
    for i in range(len(params)):
        accu_new = accu[i] + grads[i] ** 2
        accu[i] = accu_new
        temp = learning_rate * grads[i] / np.sqrt(accu_new + epsilon)
        params[i] -= temp

    return params