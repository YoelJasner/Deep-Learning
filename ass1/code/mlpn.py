import numpy as np

STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def classifier_output(x, params):
    # YOUR CODE HERE.
    # We have to remember the input every layer gets, in order to make the backpropegation
    x = np.asarray(x)
    
    for i in range(0, len(params) - 2, 2):
        x = x.dot(params[i]) + params[i+1]
        x = np.tanh(x)

    i = len(params) - 2                                 # index of last W_i
    return softmax(x.dot(params[i]) + params[i+1])      # at last layer we make softmax

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def layers_inputs(x, params):
    # returns the input vector for each layer
    inputs = []

    x = np.asarray(x)
    inputs.append(x)
    
    for i in range(0, len(params) - 2, 2):
        x = x.dot(params[i]) + params[i+1]
        x = np.tanh(x)
        inputs.append(x)        

    return inputs

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    
    y_hat = classifier_output(x, params)
    loss = -1 * np.log(y_hat[y])
    
    inputs = layers_inputs(x, params)

    grads = []
    back_pr = y_hat
    back_pr[y] -= 1

    for i in range(len(inputs) - 1, -1, -1):
        grads.insert(0, back_pr)                            # db_i
        grads.insert(0, np.outer(inputs[i], back_pr))       # dW_i
        back_pr = params[2*i].dot(back_pr)
        dtanh = (1 - np.square(inputs[i]))
        back_pr = back_pr * dtanh
    return loss, grads

def epsilon(n, m):
    return np.sqrt(6) / np.sqrt(n+m)

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        params.append(np.random.uniform(-1*epsilon(dims[i], dims[i+1]), epsilon(dims[i], dims[i+1]), [dims[i], dims[i+1]])) # W_i
        params.append(np.random.uniform(-1*epsilon(1, dims[i+1]), epsilon(1, dims[i+1]), [dims[i+1]]))                      # b_i
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    dims = [5, 4, 7, 5, 8, 6, 10, 20]
    params = create_classifier(dims)


    def _loss_and_p_grad(p):
        """
        General function - return loss and the gradients with respect to parameter p
        """
        par_num = 0                         # numer of parameter for which calculating the gradient
        for i in range(len(params)):
            if p.shape == params[i].shape:
                params[i] = p
                par_num = i

        loss, grads = loss_and_gradients(np.array(range(dims[0])), 0, params)
        return loss, grads[par_num]


    for _ in xrange(10):
        my_params = create_classifier(dims)
        for p in my_params:
            gradient_check(_loss_and_p_grad, p)
