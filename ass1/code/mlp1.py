import numpy as np

STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

def tanh(x):    
    return (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

def classifier_output(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    # YOUR CODE HERE.
    W, b, U, b_tag = params
    x = np.asarray(x)
    h = np.tanh(x.dot(W) + b)
    probs = softmax(h.dot(U) + b_tag)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    y_hat = classifier_output(x, params)

    loss = -1 * np.log(y_hat[y])

    h = np.tanh(x.dot(W) + b)                                       # h = tanh(Wx+b)
    
    y_hat[y] -= 1

    gU = np.outer(h, y_hat)                                         # dL_dU = softmax(Uh+b')*h - y_i*h
    
    gb_tag = y_hat                                                  # dL_db' = softmax(Uh+b') - y_i
    
    dL_dh = U.dot(y_hat)                                            # dL_dh = U*softmax(Uh+b') - U*y_i, 7x1
    dtanh = (1 - np.square(h))                                      # derivative of tanh, 1x7     
    
    gb = dL_dh * dtanh
    gW = np.outer(x, gb)
    
    return loss, [gW, gb, gU, gb_tag]
    
def epsilon(n, m):
    return np.sqrt(6) / np.sqrt(n+m)

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    params.append(np.random.uniform(-1*epsilon(in_dim, hid_dim), epsilon(in_dim, hid_dim), [in_dim, hid_dim]))      # W
    params.append(np.random.uniform(-1*epsilon(1, hid_dim), epsilon(1, hid_dim), [hid_dim]))                        # b
    params.append(np.random.uniform(-1*epsilon(hid_dim, out_dim), epsilon(hid_dim, out_dim), [hid_dim, out_dim]))   # U
    params.append(np.random.uniform(-1*epsilon(1, out_dim), epsilon(1, out_dim), [out_dim]))                        # b_tag

    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3, 7, 9)


    def _loss_and_U_grad(U):
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[2]


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[1]


    def _loss_and_b_tag_grad(b_tag):
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in xrange(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])

        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_b_tag_grad, b_tag)