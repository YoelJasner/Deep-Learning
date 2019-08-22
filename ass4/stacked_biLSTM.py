STUDENT = {'name': 'STEVE GUTFREUND_YOEL JASNER',
           'ID': '342873791_204380992'}

import dynet as dy
import numpy as np
import time as t
import yaml
import random
import json

UNKNOWN = "<UNK>"
PADDING = "<END>"

LABELS = ['contradiction', 'neutral', 'entailment']
L2I = {l:i for i,l in enumerate(LABELS)}
I2L = {i:l for l,i in L2I.items()}

r = open("results.txt", 'w')


class MyNetwork:
    def __init__(self, pc, vocab_size, word_emb_size, WORD_EMBEDDING=None, load=False):
        self.pc = pc

        # word embedding matrix
        self.E = self.pc.add_lookup_parameters((vocab_size, word_emb_size))
        if WORD_EMBEDDING is not None:
            self.E.init_from_array(np.array(WORD_EMBEDDING))

        # first layer of biLSTM
        self.fwdRNN1 = dy.LSTMBuilder(1, word_emb_size, 512, self.pc)
        self.bwdRNN1 = dy.LSTMBuilder(1, word_emb_size, 512, self.pc)
        
        # second layer of biLSTM
        self.fwdRNN2 = dy.LSTMBuilder(1, 2 * 512 + word_emb_size, 1024, self.pc)     # input layer is multiplied by two because we concat
        self.bwdRNN2 = dy.LSTMBuilder(1, 2 * 512 + word_emb_size, 1024, self.pc)     # the outputs of the biLSTM of the previous layer

        # third layer of biLSTM
        self.fwdRNN3 = dy.LSTMBuilder(1, 2 * 1024 + 2 * 512 + word_emb_size, 2048, self.pc)     # input layer is multiplied by two because we concat
        self.bwdRNN3 = dy.LSTMBuilder(1, 2 * 1024 + 2 * 512 + word_emb_size, 2048, self.pc)     # the outputs of the biLSTM of the previous layer

        self.W1 = self.pc.add_parameters((1600, 4 * 2 * 2048))  # multiplied by two as before
        self.b1 = self.pc.add_parameters(1600)

        self.W2 = self.pc.add_parameters((800, 1600))  # multiplied by two as before
        self.b2 = self.pc.add_parameters(800)

        self.W3 = self.pc.add_parameters((3, 800))  # multiplied by two as before
        self.b3 = self.pc.add_parameters(3)

        self.dropout_rate = 0.1

        if load:
            self.load()

    def __call__(self, inputs):
        # assuming the inputs are vectors (indexes to embedding layers)

        #dy.renew_cg()

        # word embedding for premise
        x_premise = [self.E[i] for i in inputs[0]]

        # word embedding for hypothesis
        x_hypothesis = [self.E[i] for i in inputs[1]]

        v_premise = self.sentence_to_vector(x_premise)
        v_hypothesis = self.sentence_to_vector(x_hypothesis)

        concat = dy.concatenate([v_premise, v_hypothesis])
        distance = dy.abs(v_premise - v_hypothesis)
        product = dy.cmult(v_premise, v_hypothesis)

        v = dy.concatenate([concat, distance, product])

        v = dy.rectify(self.W1 * v + self.b1)
        v = dy.dropout(v, self.dropout_rate)
        v = dy.rectify(self.W2 * v + self.b2)
        v = dy.dropout(v, self.dropout_rate)
        return dy.softmax(self.W3 * v + self.b3)

    def sentence_to_vector(self, inputs):
        f_init = self.fwdRNN1.initial_state()
        b_init = self.bwdRNN1.initial_state()
        fw_exps = f_init.transduce(inputs)
        bw_exps = b_init.transduce(inputs[::-1])        # reversing the inputs for backward lstm

        bi = [dy.concatenate([i, f, b]) for i,f, b in zip(inputs, fw_exps, bw_exps[::-1])]

        f_init = self.fwdRNN2.initial_state()
        b_init = self.bwdRNN2.initial_state()
        fw_exps = f_init.transduce(bi)
        bw_exps = b_init.transduce(bi[::-1])            # reversing the inputs for backward lstm

        bi = [dy.concatenate([i, f, b]) for i, f, b in zip(bi, fw_exps, bw_exps[::-1])]

        f_init = self.fwdRNN3.initial_state()
        b_init = self.bwdRNN3.initial_state()
        fw_exps = f_init.transduce(bi)
        bw_exps = b_init.transduce(bi[::-1])            # reversing the inputs for backward lstm

        bi = [dy.concatenate([f, b]) for f, b in zip(fw_exps, bw_exps[::-1])]
        
        a = None
        for i in range(2 * 2048):
            if a is None:
                a = dy.emax([dy.pick(b,i) for b in bi])
            else:
                a = dy.concatenate([a, dy.emax([dy.pick(b,i) for b in bi])])

        return a

    def create_network_pred_loss(self, inputs, expected_answer):
        probs = self(inputs)
        loss = -dy.log(dy.pick(probs, expected_answer))
        return probs, loss

    def create_network_return_prediction(self, inputs):
        return self(inputs)

    def save(self, i):
        self.pc.save("ttmp{}.model".format(i))

    def load(self, i):
        self.pc.populate("ttmp{}.model".format(i))


def load_data(json_file, data_type):
    print("\nLoading {} file...".format(data_type))
    start = t.time()

    data = []

    for line in open(json_file):
        js = yaml.safe_load(line)
        if js["gold_label"] == '-':
            continue
        data.append( ( (js["sentence1"],js["sentence2"]), js["gold_label"] ) )
    
    print("{} pairs in {} seconds".format(len(data), t.time() - start))
    return data


def load_pre_trained_vectors(filename):
    """
    Reads the words and their vector representations from a GloVe file.
    :param filename: Name of file to read
    :return: An embedding matrix and a dictionary that maps a word to a row-index in the matrix
    """
    print("\nLoading pre-trained vectors...")
    start = t.time()

    W2I = {}
    EMBEDDING = []

    for line in open(filename):
        l = line.split()
        W2I[l[0]] = len(W2I)
        EMBEDDING.append(np.array(map(float, l[1:])))

    # adding an embedding layer for unknown words
    EMBEDDING.append([0.0 for _ in range(300)])
    W2I[UNKNOWN] = len(W2I)

    # adding an embedding layer for unknown words
    EMBEDDING.append([0.0 for _ in range(300)])
    W2I[PADDING] = len(W2I)

    print("{} pre-trained vectors in {} seconds".format(len(W2I), t.time() - start))

    return W2I, EMBEDDING


def split_sentence(s):
    ss = []
    for p in s.split():
        if p[-1] == '.' or p[-1] == ',' or p == '!' or p == '?':
            ss.append(p[:-1])
            ss.append(p[-1])
        else:
            ss.append(p)
    return ss


def pad_with(a, pad_size, element):
    b = [element for _ in range(pad_size)]
    b[:len(a)] = a
    return b


def repr(sentence, pad=0):
    s = split_sentence(sentence)
    #s = pad_with(s, pad, PADDING)
    # replacing all the words
    x = [W2I.get(w, W2I[UNKNOWN]) for w in s]

    return x


def loss_accuracy_on_dataset(dataset, net):
    total_loss = 0.0
    good = bad = 0.0
    for inputs, label in dataset:
        premise = repr(inputs[0])
        hypothesis = repr(inputs[1])
        l = L2I[label]
        probs, loss = net.create_network_pred_loss( (premise , hypothesis), l)
        total_loss += loss.value()
        pred = np.argmax(probs.npvalue())
        good += l == pred
        bad += l != pred
    return total_loss / len(dataset), good / (good + bad)


def batched(iterable, n = 1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def find_longest(batch):
    longest = 0
    for (x,y),_ in batch:
        for s in (x,y):
            p = split_sentence(s)
            longest = max(longest, len(p))

    return longest


def train(epochs, TRAIN, net, trainer, DEV, batch_size=32):
    dev_losses = []
    dev_accus = []
    epoch_accus = []
    r.write("Training...\nThis may take a while\n\n")
    r.write("+---+------------+------------+----------+---------------\n")
    r.write("| i |  dev_loss  |  dev_accu  |   time   |  pairs seen\n")
    r.write("+---+------------+------------+----------+---------------\n")
    print("\nTraining...\nThis may take a while\n")
    print("+---+------------+------------+----------+---------------")
    print("| i |  dev_loss  |  dev_accu  |   time   |  pairs seen")
    print("+---+------------+------------+----------+---------------")
    pairs_seen = 0
    start = t.time()
    t_0 = t.time()
    for I in range(epochs):
        # random.shuffle(TRAIN)
        if I > 0 and I % 2 == 0:
            trainer.learning_rate *= 0.5
        losses = []

        dy.renew_cg()
        for line in open(TRAIN):
            js = yaml.safe_load(line)
            if js["gold_label"] == '-':
                continue
            pairs_seen += 1

            premise = repr(js["sentence1"])
            hypothesis = repr(js["sentence2"])
            label = js["gold_label"]
            l = L2I[label]
            _, loss = net.create_network_pred_loss((premise, hypothesis), l)
            losses.append(loss)

            if len(losses) == batch_size:
                batch_loss = dy.esum(losses) / batch_size
                batch_loss.backward()
                trainer.update()
                losses = []
                dy.renew_cg()

        if len(losses) < batch_size:
            batch_loss = dy.esum(losses) / batch_size
            batch_loss.backward()
            trainer.update()
        net.save(I + 1)

        dev_loss, dev_accuracy = loss_accuracy_on_dataset(DEV, net)
        dev_losses.append(dev_loss)
        dev_accus.append(dev_accuracy)
        epoch_accus.append(dev_accuracy)
        r.write("|{:2d} |  {:8.6f}  |  {:9.5f} |  {:7.2f} |  {}\n".format(I + 1, dev_loss, 100 * dev_accuracy,
                                                                          t.time() - t_0, pairs_seen))
        r.write("+---+------------+------------+----------+---------------\n")
        print("|{:2d} |  {:8.6f}  |  {:9.5f} |  {:7.2f} |  {}".format(I + 1, dev_loss, 100 * dev_accuracy,
                                                                      t.time() - t_0, pairs_seen))
        print("+---+------------+------------+----------+---------------")

    print("\nFinished training in {:5.2f}s\n".format(t.time() - start))
    r.write("\nFinished training in {:5.2f}s".format(t.time() - start))
    return dev_accus, dev_losses


if __name__ == "__main__":
    version = ""

    vectors_file = "pretrained_vectors.txt"
    data_files = "snli_1.0_{}{}.jsonl"

    W2I, WORD_EMBEDDING = load_pre_trained_vectors(vectors_file)
    # TRAIN = load_data(data_files.format("train", version), "TRAIN")
    TRAIN = data_files.format("train", version)
    DEV = load_data(data_files.format("dev", version), "DEV")

    learning_rate = 0.0002

    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(pc, alpha=learning_rate)
    net = MyNetwork(pc, len(W2I), len(WORD_EMBEDDING[0]), WORD_EMBEDDING)

    dev_accus, dev_losses = train(5, TRAIN, net, trainer, DEV, batch_size=32)

    results = {
        "accus": dev_accus,
        "losses": dev_losses,
    }
    # writing the training results
    with open("results2.json", 'a') as j:
        json.dump(results, j)
        j.write('\n')

    TEST = load_data(data_files.format("test"), "TEST")
    # TEST = data_files.format("test")
    _, test_accuracy = loss_accuracy_on_dataset(TEST, net)
    print("\n\n\t{}".format(test_accuracy))
    r.write("\n\n\ttest accuracy: {.2f}%".format(100 * test_accuracy))
    r.close()




