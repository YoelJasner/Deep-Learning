import BiLSTM_max_pooling as m
import dynet as dy
import json
import numpy as np

UNKNOWN = "<UNK>"
LABELS = ['contradiction', 'neutral', 'entailment']
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

def split_sentence(s):
    ss = []
    for p in s.split():
        if p[-1] in ['.', ',', '?', '!', ':']:
            ss.append(p[:-1])
            ss.append(p[-1])
        else:
            ss.append(p)
    return ss


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


if __name__ == '__main__':
    losses = []
    accus = []

    version = "_plan_B"

    vectors_file = "pretrained_vectors.txt"
    data_file = "snli_1.0_{}{}.jsonl"

    DEV = m.load_data(data_file.format("dev", version), "DEV")

    W2I, WORD_EMBEDDING = m.load_pre_trained_vectors(vectors_file)

    r = open("dev_results.txt", 'w')
    print("\nepoch\tloss\taccuracy\n")
    r.write("\nepoch\tloss\taccuracy\n")

    pc = dy.ParameterCollection()
    net = m.MyNetwork(pc, len(W2I), len(WORD_EMBEDDING[0]), WORD_EMBEDDING)
    net.load(5)
    TEST = m.load_data(data_file.format("test", version), "TEST")
    test_loss, test_accu = loss_accuracy_on_dataset(TEST, net)
    print(test_accu)
    '''
    for i in range(1):
        pc = dy.ParameterCollection()
        net = m.MyNetwork(pc, len(W2I), len(WORD_EMBEDDING[0]), WORD_EMBEDDING)
        net.load(i+1)

        loss, accu = loss_accuracy_on_dataset(DEV, net)
        losses.append(loss)
        accus.append(accu)
        print("{}\t{}\t{}".format(i+1, loss, accu))
        r.write("{}\t{}\t{}\n".format(i+1, loss, accu))

        if i == 9:
            TEST = m.load_data(data_file.format("test", version), "TEST")
            test_loss, test_accu = loss_accuracy_on_dataset(TEST, net)
    '''
    r.close()

    results = {
        "accus": accus,
        "losses": losses,
    }
    # writing the training results
    with open("results.json", 'a') as j:
        json.dump(results, j)
        j.write('\n')
