import random
import numpy as np
import mlp1 as mlp
import xor_data as xr

STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for y, x in dataset:
        y_hat = mlp.predict(x, params)
        good += y_hat == y
        bad += y_hat != y

    return good / (good + bad)


if __name__ == "__main__":

    num_iterations = 0
    dev_acc = 0

    learning_rate = 0.1
    train_data = [(l, np.array(f)) for l, f in xr.data]
    dev_data = list(train_data)
    in_dim = 2
    out_dim = 2
    hid_dim = 10

    params = mlp.create_classifier(in_dim, hid_dim, out_dim)

    while dev_acc < 1.0:
        num_iterations += 1                     # counting the iterations
        cum_loss = 0.0
        random.shuffle(train_data)
        for y, x in train_data:
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            
            for i in range(4):
                params[i] -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_acc = accuracy_on_dataset(dev_data, params)
        print num_iterations, train_loss, train_accuracy, dev_acc

    print "(0,0) => ", mlp.predict([0, 0], params)
    print "(0,1) => ", mlp.predict([0, 1], params)
    print "(1,0) => ", mlp.predict([1, 0], params)
    print "(1,1) => ", mlp.predict([1, 1], params)
