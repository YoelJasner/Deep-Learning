STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}
         
import dynet as dy
import random
import numpy as np
import time as t


C2I = {c:i for i,c in enumerate(list("123456789abcd"))}
L2I = {l:i for i,l in enumerate(["good", "bad"])}
I2L = {i:l for l,i in L2I.iteritems()}

sum_C2I = {c:i for i,c in enumerate(list("0123456789#"))}
palindrome_C2I = {c:i for i,c in enumerate(list("123456789abcdefghijklmnopqrstuvwxyz#"))}
mul_C2I = {c:i for i,c in enumerate(list("0123456789#"))}

class MyNetwork:
    def __init__(self, pc, vocab_size, char_emb_size, in_dim, hid_dim, out_dim):
        # Embedding matrix
        self.E = pc.add_lookup_parameters((vocab_size, char_emb_size))

        # LSTM
        self.lstm = dy.LSTMBuilder(1, char_emb_size, in_dim, pc)
        
        # parameters for hidden layer
        self.W1 = pc.add_parameters((hid_dim, in_dim))
        self.b1 = pc.add_parameters((hid_dim))

        self.W2 = pc.add_parameters((out_dim, hid_dim))
        self.b2 = pc.add_parameters((out_dim))

    def __call__(self, inputs):
        '''
        returns a vector of probabilities
        '''
        state = self.lstm.initial_state()
        x_t = [self.E[i] for i in inputs]
        outputs = state.transduce(x_t)
        # we consider only the output of the last LSTM unit
        y_t = outputs[-1]
        return dy.softmax(self.W2 * dy.tanh((self.W1 * y_t) + self.b1) + self.b2)
    
    def create_network_pred_loss(self, inputs, expected_answer):
        dy.renew_cg() # new computation graph
        out = self(inputs)
        out.value()
        pred = np.argmax(out.npvalue())
        loss =  -dy.log(dy.pick(out, expected_answer))
        return pred, loss
    
    def create_network_return_prediction(self, inputs):
        dy.renew_cg() # new computation graph
        out = self(inputs)
        return np.argmax(out.npvalue())    

def predict(data, filename, net):
    with open(filename, "w") as f:
        for inputs in data:
            x = [C2I[c] for c in inputs]
            y_hat = net.create_network_return_prediction(x)
            label = I2L[y_hat]
            f.write('{0} {1}\n'.format(inputs, label))


def loss_accuracy_on_dataset(dataset, net):
    total_loss = 0.0 # total loss in this iteration.
    good = bad = 0.0        
    for inputs, label in dataset:
        x = [C2I[c] for c in inputs]
        y = L2I[label]
        y_hat, loss = net.create_network_pred_loss(x, y)
        total_loss += loss.value()
        good += y_hat == y
        bad += y_hat != y
    
    return total_loss / len(dataset), good / (good + bad)


def train(epochs, TRAIN, net, trainer, DEV, r=None):
    dev_losses = []
    dev_accus = []
    r.write("\n\nTraining...\nThis may take a while\n\n")
    r.write("+---+------------+------------+------------+--------+\n")
    r.write("| i | train_loss |  dev_loss  |  dev_accu  |  time  |\n")
    r.write("+---+------------+------------+------------+--------+\n")
    print("\nTraining...\nThis may take a while\n")
    print("+---+------------+------------+------------+--------+")
    print("| i | train_loss |  dev_loss  |  dev_accu  |  time  |")
    print("+---+------------+------------+------------+--------+")
    start = t.time()
    for I in range(epochs):
        t_0 = t.time()
        total_loss = 0.0 # total loss in this iteration.
        random.shuffle(TRAIN)
        total_loss = 0.0
        for inputs, label in TRAIN:
            x = [C2I[c] for c in inputs]
            y = L2I[label]
            _, loss = net.create_network_pred_loss(x, y)
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            
        train_loss = total_loss / len(TRAIN)
        dev_loss, dev_accuracy = loss_accuracy_on_dataset(DEV, net)
        dev_losses.append(dev_loss)
        dev_accus.append(dev_accuracy)
        r.write("|{:2d} |  {:8.6f}  |  {:8.6f}  |  {:9.5f} |  {:5.2f} |\n".format(I+1, train_loss, dev_loss, 100 * dev_accuracy, t.time() - t_0))
        r.write("+---+------------+------------+------------+--------+\n")
        print("|{:2d} |  {:8.6f}  |  {:8.6f}  |  {:9.5f} |  {:5.2f} |".format(I+1, train_loss, dev_loss, 100 * dev_accuracy, t.time() - t_0))
        print("+---+------------+------------+------------+--------+")
    print("\nFinished training in {:5.2f}s".format(t.time() - start))
    r.write("\nFinished training in {:5.2f}s".format(t.time() - start))
    return dev_accus, dev_losses

def read_data(filename, labels=True):
    data = []
    for l in open(filename):
        if labels:
            x,y = l.strip().split()
            data.append([x,y])
        else:
            x = l.strip()
            data.append(x)
    return data

if __name__ == "__main__":
    word_emb_size = 50
    hid_dim = 50
    in_dim = 100 
    epochs = 5

    language = ''
    
    if language == "sum_":
        C2I = sum_C2I
        epochs = 100
    if language == "palindrome_":
        C2I = palindrome_C2I
        epochs = 100
    if language == "mul_":
        C2I = mul_C2I
        epochs = 100

    TRAIN = read_data(language + "train")
    DEV = read_data(language + "dev")
    TEST = read_data(language + "test", False)

    r = open("result", 'w')
    r.write("training set:\t\t{0}\ndev set:\t\t{1}\ntest set:\t\t{2}\nepochs:\t\t\t{3}\nLSTM output:\t\t{4}\nhidden layer:\t\t{5}"
            .format(len(TRAIN), len(DEV), len(TEST), epochs, in_dim, hid_dim))
    print ("training set:\t\t{0}\ndev set:\t\t{1}\ntest set:\t\t{2}\nepochs:\t\t\t{3}\nLSTM output:\t\t{4}\nhidden layer:\t\t{5}"
            .format(len(TRAIN), len(DEV), len(TEST), epochs, in_dim, hid_dim))
    
    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(pc)
    net = MyNetwork(pc, len(C2I), word_emb_size, in_dim, hid_dim, len(L2I))

    dev_acc, dev_loss = train(epochs, TRAIN, net, trainer, DEV, r)
    r.close()

    dev_acc = [100 * a for a in dev_acc]

    outfile = "test.pred"
    predict(TEST, outfile, net)