STUDENT = {'name': 'STEVE GUTFREUND_YOEL JASNER',
           'ID': '342873791_204380992'}

import linecache
import yaml
import time


W2I = {}
INDEXES = set()


def split_sentence(s):
    ss = []
    for p in s.split():
        if p[-1] in ['.', ',', '?', '!', ':']:
            ss.append(p[:-1])
            ss.append(p[-1])
        else:
            ss.append(p)
    return ss


def get_indexes(filename):
    global INDEXES
    for line in open(filename):
        js = yaml.safe_load(line)
        if js["gold_label"] == '-':
            continue

        for sentence in ["sentence1", "sentence2"]:
            s = split_sentence(js[sentence])
            # replacing all the words to indexes and saving them in a set (to avoid duplicates)
            INDEXES.update([W2I.get(w, -1) for w in s])


def main():
    global W2I

    start = time.time()
    print("\nLoading pre-trained vectors...")
    W2I = {}

    n = 0
    for line in open("glove.840B.300d.txt"):
        n += 1
        l = line.split()
        W2I[l[0]] = n

    print("{} pre-trained vectors in {} seconds".format(len(W2I), time.time() - start))

    start = time.time()
    print("\nWriting pre-trained vectors for our datasets...")

    print("getting indexes for train set")
    get_indexes("snli_1.0_train.jsonl")
    print("getting indexes for dev set")
    get_indexes("snli_1.0_dev.jsonl")
    print("getting indexes for test set")
    get_indexes("snli_1.0_test.jsonl")

    t = open("pretrained_vectors.txt", 'w')
    for i in INDEXES:
        if i == -1:
            continue
        t.write(linecache.getline("glove.840B.300d.txt", i))

    print("wrote pre-trained vectors in {} seconds".format(time.time() - start))


if __name__ == '__main__':
    main()
