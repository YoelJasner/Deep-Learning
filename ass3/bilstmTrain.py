STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

from sys import argv
import bilstm_model as bilstm
import dynet as dy
import random
import time as t
import json

L2I = {}
I2L = {}
X2I = {}
UNKNOWN = "<UNK>"
WORDS = set()

def loss_accuracy_on_dataset(dataset, net):
    total_loss = 0.0 # total loss in this iteration.
    good = bad = 0.0        
    for inputs, labels in dataset:
            y = [L2I[lbl] for lbl in labels]
            y_hat, loss = net.create_network_pred_loss(inputs, y)
            total_loss += dy.esum(loss).value()
            for i,j in zip(y, y_hat):
                if TASK == 'ner' and i == j == L2I['O']:
                    continue
                good += i == j
                bad += i != j
    
    return total_loss / len(dataset), good / (good + bad)

def train(epochs, TRAIN, net, trainer, DEV, r=None):
    dev_losses = []
    dev_accus = []
    r.write("Training...\nThis may take a while\n\n")
    r.write("+---+------------+------------+-----------+--------+\n")
    r.write("| i |  dev_loss  |  dev_accu  | sentences |  time  |\n")
    r.write("+---+------------+------------+-----------+--------+\n")
    print("\nTraining...\nThis may take a while\n")
    print("+---+------------+------------+-----------+--------+")
    print("| i |  dev_loss  |  dev_accu  | sentences |  time  |")
    print("+---+------------+------------+-----------+--------+")
    start = t.time()
    sentences_seen = 0
    sentences_parts = []
    t_0 = t.time()
    for I in range(epochs):
        total_loss = 0.0 # total loss in this iteration.
        random.shuffle(TRAIN)
        total_loss = 0.0
        for inputs, labels in TRAIN:
            sentences_seen += 1
            l = [L2I[lbl] for lbl in labels]
            _, loss = net.create_network_pred_loss(inputs, l)
            loss = dy.esum(loss)
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            if sentences_seen % 500 == 0 or sentences_seen == epochs*len(TRAIN):
                sentences_parts.append(sentences_seen)
                dev_loss, dev_accuracy = loss_accuracy_on_dataset(DEV, net)
                dev_losses.append(dev_loss)
                dev_accus.append(dev_accuracy)
                r.write("|{:2d} |  {:8.6f}  |  {:9.5f} |   {:5d}   |  {:5.2f} |\n".format(I+1, dev_loss, 100 * dev_accuracy, sentences_seen, t.time() - t_0))
                r.write("+---+------------+------------+-----------+--------+\n")
                print("|{:2d} |  {:8.6f}  |  {:9.5f} |   {:5d}   |  {:5.2f} |".format(I+1, dev_loss, 100 * dev_accuracy, sentences_seen, t.time() - t_0))
                print("+---+------------+------------+-----------+--------+")
                t_0 = t.time()

            
        #train_loss = total_loss / len(TRAIN)        
    print("\nFinished training in {:5.2f}s\n".format(t.time() - start))
    r.write("\nFinished training in {:5.2f}s".format(t.time() - start))
    return dev_accus, dev_losses, sentences_parts

def split_word(word):
    if len(word) <= 3:
        return [word]
    
    return [word[:3], word, word[-3:]]

def get_data(filename, update=True, sub_words=True):
    global WORDS,L2I,I2L
    data = []
    sentence = []
    labels = []
    for line in open(filename):
        if line == '\n':
            data.append([sentence, labels])
            sentence = []
            labels = []
            continue
        w,l = line.strip().split()
        sentence.append(w)
        labels.append(l)
        if update:
            if l not in L2I:
                L2I[l] = len(L2I)
                I2L[L2I[l]] = l
            WORDS.add(w)
            if sub_words:
                for part in split_word(w):
                    WORDS.add(part)

    return data

def main():
    global X2I,L2I,I2L,WORDS, TASK
    embed_dim = 64
    repr_dim = 64
    lstm_layers = 1
    lstm_dim1 = 32
    lstm_dim2 = 32
    epochs = 5
    learning_rate = 0.0005

    if len(argv) != 6:
        print("Wrong command-line arguments!")
        exit()

    # parameters passed by the command-line
    repr_c = argv[1]
    trainfile = argv[2] 
    modelfile = argv[3]
    devfile = argv[4]
    TASK = argv[5]

    subs = False
    if repr_c == "c":
        subs = True

    r = open('result', 'w')
    print("\ntask:\t\t\t{0}\nepochs:\t\t\t{1}\nembed_dim:\t\t{2}\nrepr_dim:\t\t{3}\nlstm_dim1:\t\t{4}\nlstm_dim2:\t\t{5}\nrepresentor:\t\t{6}\nlearning_rate:\t\t{7}"
        .format(TASK, epochs, embed_dim, repr_dim, lstm_dim1, lstm_dim2, repr_c, learning_rate))
    r.write("task:\t\t\t{0}\nepochs:\t\t\t{1}\nembed_dim:\t\t{2}\nrepr_dim:\t\t{3}\nlstm_dim1:\t\t{4}\nlstm_dim2:\t\t{5}\nrepresentor:\t\t{6}\n\nlearning_rate:\t\t{7}\n"
        .format(TASK, epochs, embed_dim, repr_dim, lstm_dim1, lstm_dim2, repr_c, learning_rate))

    TRAIN = get_data(trainfile, update=True, sub_words=subs)
    DEV = get_data(devfile, update=False, sub_words=subs)
    
    data = {
        "WORDS": list(WORDS),
        "L2I": L2I,
        "embed_dim": 64,
        "repr_dim": 64,
        "lstm_layers": 1,
        "lstm_dim1": 32,
        "lstm_dim2": 32
    }

    # saving sizes of the network components
    with open(TASK + "_parameters.json", 'w') as t:
        json.dump(data, t)
    
    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(pc, alpha=learning_rate)

    # option (a)
    if repr_c == 'a':
        X2I = {w:i for i,w in enumerate(WORDS)}
        X2I[UNKNOWN] = len(X2I)                     # for the words we encounter outside the training test
        net = bilstm.WordEmbeddingNet(pc, X2I, len(X2I), embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, len(L2I))
    
    # option (b)
    if repr_c == 'b':
        for w in WORDS:
            for c in w:
                if c not in X2I:
                    X2I[c] = len(X2I)
        X2I[UNKNOWN] = len(X2I)                     # for the words we encounter outside the training test
        net = bilstm.CharEmbeddedLSTMNet(pc, X2I, len(X2I), embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, len(L2I))

    # option (c)
    if repr_c == 'c':
        X2I = {w:i for i,w in enumerate(WORDS)}
        X2I[UNKNOWN] = len(X2I)                     # for the words we encounter outside the training test
        net = bilstm.SubwordEmbeddedNet(pc, X2I, len(X2I), embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, len(L2I))

    # option (d)
    if repr_c == 'd':
        X2I = {w:i for i,w in enumerate(WORDS)}
        for w in WORDS:
            for c in w:
                if c not in X2I:
                    X2I[c] = len(X2I)
        X2I[UNKNOWN] = len(X2I)                     # for the words we encounter outside the training test
        net = bilstm.WordCharEmbeddedNet(pc, X2I, len(X2I), embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, len(L2I))
    
    dev_acc, _, sentences_parts = train(epochs, TRAIN, net, trainer, DEV, r)
    r.close()

    # saving the model for later use
    net.save_to(modelfile)

    dev_acc = [100 * a for a in dev_acc]

    results = {
        "name": modelfile,
        "dev_acc": dev_acc,
        "sentences": sentences_parts
    }

    # writing the training results
    with open(TASK + "_dev_results.json", 'a') as r:
        json.dump(results, r)
        r.write('\n')


if __name__ == "__main__":
    main()
