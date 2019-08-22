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
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

r = open("results.txt", 'w')


class MyNetwork:
    def __init__(self, pc, vocab_size, word_emb_size, WORD_EMBEDDING=None):
        self.pc = pc

        # word embedding matrix
        self.E = self.pc.add_lookup_parameters((vocab_size, word_emb_size))
        if WORD_EMBEDDING is not None:
            self.E.init_from_array(np.array(WORD_EMBEDDING))

        # first layer of biLSTM
        self.fwdRNN = dy.LSTMBuilder(1, word_emb_size, 512, self.pc)
        self.bwdRNN = dy.LSTMBuilder(1, word_emb_size, 512, self.pc)

        self.W1 = self.pc.add_parameters((512, 4 * 2 * 512))  # multiplied by two for bi-directional RNN
        self.b1 = self.pc.add_parameters(512)

        self.W2 = self.pc.add_parameters((3, 512))
        self.b2 = self.pc.add_parameters(3)


    def __call__(self, inputs):
        # assuming the inputs are vectors (indexes to embedding layers)

        # word embedding for premise
        x_premise = [dy.lookup(self.E, i, update=False) for i in inputs[0]]

        # word embedding for hypothesis
        x_hypothesis = [dy.lookup(self.E, i, update=False) for i in inputs[0]]

        v_premise = self.sentence_to_vector(x_premise)
        v_hypothesis = self.sentence_to_vector(x_hypothesis)

        concat = dy.concatenate([v_premise, v_hypothesis])
        distance = dy.abs(v_premise - v_hypothesis)
        product = dy.cmult(v_premise, v_hypothesis)

        v = dy.concatenate([concat, distance, product])

        v = dy.rectify(self.W1 * v + self.b1)
        return dy.softmax(self.W2 * v + self.b2)

    def sentence_to_vector(self, inputs):
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()
        fw_exps = f_init.transduce(inputs)
        bw_exps = b_init.transduce(inputs[::-1])  # reversing the inputs for backward lstm

        bi = [dy.concatenate([i, f, b]) for i, f, b in zip(inputs, fw_exps, bw_exps[::-1])]

        a = dy.concatenate([dy.emax([dy.pick(b, i) for b in bi]) for i in range(2 *512)])

        return a

    def create_network_pred_loss(self, inputs, expected_answer):
        probs = self(inputs)
        loss = -dy.log(dy.pick(probs, expected_answer))
        return probs, loss

    def create_network_return_prediction(self, inputs):
        return self(inputs)

    def save(self, i):
        self.pc.save("tmp{}.model".format(i))

    def load(self, i):
        self.pc.populate("tmp{}.model".format(i))


def load_data(json_file, data_type):
    print("\nLoading {} file...".format(data_type))
    start = t.time()

    data = []

    for line in open(json_file):
        js = yaml.safe_load(line)
        if js["gold_label"] == '-':
            continue
        data.append(((js["sentence1"], js["sentence2"]), js["gold_label"]))

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
        if p[-1] in ['.', ',', '?', '!', ':']:
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
    # replacing all the words
    x = [W2I.get(w, W2I[UNKNOWN]) for w in split_sentence(sentence)]

    return x


def loss_accuracy_on_dataset(dataset, net):
    total_loss = 0.0
    good = bad = 0.0
    for inputs, label in dataset:
        dy.renew_cg()
        premise = repr(inputs[0])
        hypothesis = repr(inputs[1])
        l = L2I[label]
        probs, loss = net.create_network_pred_loss((premise, hypothesis), l)
        total_loss += loss.value()
        pred = np.argmax(probs.npvalue())
        good += l == pred
        bad += l != pred

    return total_loss / (good + bad), good / (good + bad)


def batched(iterable, n=1):
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
    for (x, y), _ in batch:
        for s in (x, y):
            p = split_sentence(s)
            longest = max(longest, len(p))

    return longest


def train(epochs, TRAIN, net, trainer, DEV, batch_size=32):
    epoch_accus = []
    dev_losses = []
    dev_accus = []
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
        #random.shuffle(TRAIN)
        if I > 0 and I % 2 == 0:
            trainer.learning_rate *= 0.2
        losses = []

        dy.renew_cg()
        for line in open(TRAIN):
            js = yaml.safe_load(line)
            if js["gold_label"] == '-':
                continue
            pairs_seen += 1
            if pairs_seen % 1000 == 0:
                print(pairs_seen)
            
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
        
        net.save(I+1)

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

    version = "_plan_B"

    vectors_file = "pretrained_vectors.txt"
    data_files = "snli_1.0_{}{}.jsonl"

    W2I, WORD_EMBEDDING = load_pre_trained_vectors(vectors_file)
    #TRAIN = load_data(data_files.format("train", version), "TRAIN")
    TRAIN = data_files.format("train", version)
    DEV = load_data(data_files.format("dev", version), "DEV")

    learning_rate = 0.005

    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(pc, alpha=learning_rate)
    net = MyNetwork(pc, len(W2I), len(WORD_EMBEDDING[0]), WORD_EMBEDDING)

    dev_accus, dev_losses = train(5, TRAIN, net, trainer, DEV, batch_size=64)

    results = {
        "accus": dev_accus,
        "losses": dev_losses,
    }
    # writing the training results
    with open("results.json", 'a') as j:
        json.dump(results, j)
        j.write('\n')

    TEST = load_data(data_files.format("test", version), "TEST")
    #TEST = data_files.format("test", version)
    _, test_accuracy = loss_accuracy_on_dataset(TEST, net)
    print("\n\n\t{}".format(test_accuracy))
    r.write("\n\n\ttest accuracy: {:.2f}%".format(100 * test_accuracy))
    r.close()



