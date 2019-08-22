# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.

STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

import random
def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def get_bigram_sets():
    TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("data\\train")]
    DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("data\\dev")]
    TEST  = [(l,text_to_bigrams(t)) for l,t in read_data("data\\test")]

    from collections import Counter
    fc = Counter()
    for l,feats in TRAIN:
        fc.update(feats)

    # 600 most common bigrams in the training set.
    vocab = set([x for x,c in fc.most_common(600)])

    # label strings to IDs
    L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
    # feature strings (bigrams) to IDs
    F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

    return [TRAIN, DEV, TEST, vocab, L2I, F2I]

def text_to_unigrams(text):
    return ["%s" % (c) for c in text]

def get_unigram_sets():
    TRAIN = [(l,text_to_unigrams(t)) for l,t in read_data("data\\train")]
    DEV   = [(l,text_to_unigrams(t)) for l,t in read_data("data\\dev")]
    TEST  = [(l,text_to_unigrams(t)) for l,t in read_data("data\\test")]

    from collections import Counter
    fc = Counter()
    for l,feats in TRAIN:
        fc.update(feats)

    # 600 most common unigrams in the training set.
    vocab = set([x for x,c in fc.most_common(600)])
    
    # label strings to IDs
    L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
    # feature strings (bigrams) to IDs
    F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

    return [TRAIN, DEV, TEST, vocab, L2I, F2I]