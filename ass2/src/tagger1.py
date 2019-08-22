STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}
         
import dynet as dy
import utils as ut
import random
import numpy as np
import time as t
#import matplotlib.pyplot as plt
import sys


class MyNetwork:
    def __init__(self, pc, vocab_size, word_emb_size, window_size, hid_dim, out_dim, EMBED, sub_words=False):
        # Embedding matrix
        self.E = pc.add_lookup_parameters((vocab_size, word_emb_size))
        if EMBED != []:
            self.E.init_from_array(np.array(EMBED))

        in_dim = word_emb_size * window_size
        # parameters for hidden layer
        self.W1 = pc.add_parameters((hid_dim, in_dim))
        self.b1 = pc.add_parameters((hid_dim))

        self.W2 = pc.add_parameters((out_dim, hid_dim))
        self.b2 = pc.add_parameters((out_dim))

        self.embedding = self.embed_words
        if sub_words:
            self.embedding = self.embed_sub_words

    
    def embed_words(self, inputs):
        return dy.concatenate([self.E[i] for i in inputs])
    
    def embed_sub_words(self, inputs):
        x = []
        for part in inputs:
            x.append(dy.esum([self.E[i] for i in part]))
        return dy.concatenate([v for v in x])
    
    def __call__(self, inputs):
        '''
        returns a vector of probabilities
        '''
        x = self.embedding(inputs)
        return dy.softmax(self.W2 * dy.tanh((self.W1 * x) + self.b1) + self.b2)
    
    def create_network_pred_loss(self, inputs, expected_answer):
        dy.renew_cg() # new computation graph
        out = self(inputs)
        pred = np.argmax(out.npvalue())
        loss =  -dy.log(dy.pick(out, expected_answer))
        return pred, loss
    
    def create_network_return_prediction(self, inputs):
        dy.renew_cg() # new computation graph
        out = self(inputs)
        return np.argmax(out.npvalue())    

def predict(data, filename, net, input2vec):
    with open(filename, "w") as f:
        for inputs in data:
            if inputs == '\n':
                f.write('\n')
                continue
            x = input2vec(inputs)
            y_hat = net.create_network_return_prediction(x)
            label = ut.I2L[y_hat]
            f.write('{0} {1}\n'.format(inputs[2], label))


def loss_accuracy_on_dataset(dataset, net, input2vec):
    total_loss = 0.0 # total loss in this iteration.
    good = bad = 0.0        
    for inputs, label in dataset:
            inputs = input2vec(inputs)
            label = ut.L2I[label]
            y_hat, loss = net.create_network_pred_loss(inputs, label)
            total_loss += loss.value()
            if ut.TASK == 'ner' and y_hat == label == ut.L2I['O']:
                continue
            good += y_hat == label
            bad += y_hat != label
    
    return total_loss / len(dataset), good / (good + bad)


def train(epochs, TRAIN, net, trainer, DEV, input2vec, r):
    dev_losses = []
    dev_accus = []
    r.write("Training...\nThis may take a while\n\n")
    r.write("+---+------------+------------+------------+--------+\n")
    r.write("| i | train_loss |  dev_loss  |  dev_accu  |  time  |\n")
    r.write("+---+------------+------------+------------+--------+\n")
    print("\nTraining...\nThis may take a while\n")
    print("+---+------------+------------+------------+--------+")
    print("| i | train_loss |  dev_loss  |  dev_accu  |  time  |")
    print("+---+------------+------------+------------+--------+")
    for I in range(epochs):
        t_0 = t.time()
        total_loss = 0.0 # total loss in this iteration.
        random.shuffle(TRAIN)
        total_loss = 0.0
        for inputs, label in TRAIN:
            inputs = input2vec(inputs)
            label = ut.L2I[label]
            _, loss = net.create_network_pred_loss(inputs, label)
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            
        train_loss = total_loss / len(TRAIN)
        dev_loss, dev_accuracy = loss_accuracy_on_dataset(DEV, net, input2vec)
        dev_losses.append(dev_loss)
        dev_accus.append(dev_accuracy)
        r.write("|{:2d} |  {:8.6f}  |  {:8.6f}  |  {:8.5f}  | {:6.2f} |\n".format(I+1, train_loss, dev_loss, 100 * dev_accuracy, t.time() - t_0))
        r.write("+---+------------+------------+------------+--------+\n")
        print("|{:2d} |  {:8.6f}  |  {:8.6f}  |  {:8.5f}  | {:6.2f} |".format(I+1, train_loss, dev_loss, 100 * dev_accuracy, t.time() - t_0))
        print("+---+------------+------------+------------+--------+")
    return dev_accus, dev_losses


def convert_words_to_vec(inputs):
    if pre_trained:
        vec = []
        for w in inputs:
            if w not in ['*START*', '*END*', '*UNKNOWN*']:
                w.lower()
            vec.append(ut.W2I.get(w, ut.W2I['*UNKNOWN*']))
        return vec 
    return [ut.W2I.get(w, ut.W2I['*UNKNOWN*']) for w in inputs]

def convert_sub_words_to_vec(inputs):
    vec = []
    for word in inputs:
        vec.append(convert_words_to_vec(ut.split_word(word)))
    return vec

def input_validation():
    test = ''
    pre_trained = False
    sub_words = False
    task = ''

    if len(sys.argv) == 1:
        print("You should at least specify #test and pos/ner, for examle: python src/tagger1.py test1 ner")
        exit()
    for i in range(1, len(sys.argv)):
        if i == 1:
            test = sys.argv[1]
            if test != 'test1' and test != 'test3' and test != 'test4':
                print('Wrong test number')
                exit()
        if i == 2:
            task = sys.argv[2]
            if task != 'ner' and task != 'pos':
                print("First parameter has to be pos or ner, for examle: python src/tagger1.py ner")
                exit()
        if i == 3 or i == 4:
            if sys.argv[i] == 'pre':
                pre_trained = True
            elif sys.argv[i] == 'sub':
                sub_words = True
            else:
                print("Bad parameter")
                exit()
    return test, pre_trained, sub_words, task

if __name__ == "__main__":    
    word_emb_size = 50
    window_size = 5
    hid_dim = 150     
    learning_rate = 0.01
    epochs = 15

    directory = 'stam/'
    
    global pre_trained
    test, pre_trained, sub_words, task = input_validation()
    ut.run(task, pre_trained, sub_words)

    
    r = open(directory + "result", 'w')
    r.write("test:\t\t\t{0}\ntask:\t\t\t{1}\npre-trained embeddings:\t{2}\nsub-words:\t\t{3}".format(test, task.upper(), pre_trained, sub_words)
            + "\nepochs:\t\t\t{0}\nlearning rate:\t\t{1}\nwindow size:\t\t{2}\nhidden layer:\t\t{3}\n\n".format(epochs, learning_rate, window_size, hid_dim))
    print("\ntest:\t\t\t{0}\ntask:\t\t\t{1}\npre-trained embeddings:\t{2}\nsub-words:\t\t{3}".format(test, task.upper(), pre_trained, sub_words)
            + "\nepochs:\t\t\t{0}\nlearning rate:\t\t{1}\nwindow size:\t\t{2}\nhidden layer:\t\t{3}\n".format(epochs, learning_rate, window_size, hid_dim))

    input2vec = convert_words_to_vec
    if sub_words:
        input2vec = convert_sub_words_to_vec
    
    pc = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(pc, learning_rate)
    net = MyNetwork(pc, len(ut.W2I), word_emb_size, window_size, hid_dim, len(ut.L2I), ut.EMBED, sub_words)

    dev_acc, dev_loss = train(epochs, ut.TRAIN, net, trainer, ut.DEV, input2vec, r)
    r.close()

    dev_acc = [100 * a for a in dev_acc]

    predict(ut.TEST, directory + test + '.' + task, net, input2vec)

    '''
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), dev_acc, 'o-')
    ax.set_title('{0} - dev accuracy'.format(ut.TASK))
    ax.set_xticks(range(1, epochs+1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True)
    plt.savefig(directory + 'accu.png')
    
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), dev_loss, 'o-')
    ax.set_title('{0} - dev loss'.format(ut.TASK))
    ax.set_xticks(range(1, epochs+1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.savefig(directory + 'loss.png')
    '''