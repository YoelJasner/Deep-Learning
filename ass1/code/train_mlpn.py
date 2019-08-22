import mlpn as mlp
import random
import numpy as np
import utils as ut

STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    x = np.zeros(len(vocab))
    for f in features:
        if F2I.has_key(f): 
            x[F2I[f]] += 1
    return x

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)
        y = L2I[label]
        y_hat = mlp.predict(x, params)
        good += y_hat == y
        bad += y_hat != y
        
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)          # convert features to a vector.
            y = L2I[label]                   # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
            
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy        
    return params


def test_pred(test_data, params):
    with open("test.pred", "w") as f:
        for _, features in test_data:
            x = feats_to_vec(features)
            y_hat = mlp.predict(x, params)
            for l,i in L2I.items():
                if i == y_hat:
                    f.write('{}\n'.format(l))
                    break
                

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.    
    # ...

    uni = False
    
    if not uni:
        train_data, dev_data, test_data, vocab, L2I, F2I = ut.get_bigram_sets()
    else:
        train_data, dev_data, test_data, vocab, L2I, F2I = ut.get_unigram_sets()

    in_dim = len(vocab)
    out_dim = len(L2I)
    num_iterations = 30
    learning_rate = 0.001
    
    #params = mlp.create_classifier([in_dim, 60, 15, out_dim])
    params = mlp.create_classifier([in_dim, out_dim])    
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    test_pred(test_data, trained_params)
