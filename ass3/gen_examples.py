STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

import random as r
import os
import numpy as np
import math as m

MAX_DIGITS = 10

def create_pos_examples(amount, max_digits=MAX_DIGITS):
    '''
        returns an array of sequences in form of:
        [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
        param amount: size of array
        param max_digits: maximum amount of digits in a sequence
    '''
    data = []
    for _ in range(amount):
        digits_seq = [''.join([r.choice("123456789") for _ in range(r.randint(1, max_digits))]) for _ in range(5)]
        letters_seq = [''.join([r.choice(c) for  _ in range(r.randint(1, max_digits))]) for c in ['a', 'b', 'c', 'd']]
        data.append("{0}{1}{2}{3}{4}{5}{6}{7}{8}"
                .format(digits_seq[0], letters_seq[0], digits_seq[1], letters_seq[1], digits_seq[2], letters_seq[2], digits_seq[3], letters_seq[3], digits_seq[4]))
    return data

def create_neg_examples(amount, max_digits=MAX_DIGITS):
    '''
        returns an array of sequences in form of:
        [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
        param amount: size of array
        param max_digits: maximum amount of digits in a sequence
    '''
    data = []
    for _ in range(amount):
        digits_seq = [''.join([r.choice("123456789") for _ in range(r.randint(1, max_digits))]) for _ in range(5)]
        letters_seq = [''.join([r.choice(c) for  _ in range(r.randint(1, max_digits))]) for c in ['a', 'c', 'b', 'd']]
        data.append("{0}{1}{2}{3}{4}{5}{6}{7}{8}"
                .format(digits_seq[0], letters_seq[0], digits_seq[1], letters_seq[1], digits_seq[2], letters_seq[2], digits_seq[3], letters_seq[3], digits_seq[4]))
    return data


def create_data_with_labels(amount, max_digits=MAX_DIGITS):
    data = ["{0} good".format(seq) for seq in create_pos_examples(int(m.ceil(amount / 2)), max_digits)]
    data.extend(["{0} bad".format(seq) for seq in create_neg_examples(int(m.floor(amount / 2)), max_digits)])
    r.shuffle(data)
    return data

def create_data_without_labels(amount, max_digits=MAX_DIGITS):
    data = ["{0}".format(seq) for seq in create_pos_examples(int(m.ceil(amount / 2)), max_digits)]
    data.extend(["{0}".format(seq) for seq in create_neg_examples(int(m.floor(amount / 2)), max_digits)])
    r.shuffle(data)
    return data

def create_test_files(amount, max_digits=MAX_DIGITS):
    data_labeled = create_data_with_labels(amount, max_digits)
    data_not_labeled = [l.split()[0] for l in data_labeled]
    return data_not_labeled, data_labeled
    

def main():
    pos = open('pos_examples', 'w')
    neg = open('neg_examples', 'w')
    train = open('train', 'w')
    dev = open('dev', 'w')
    test = open('test', 'w')
    test_labeled = open('test_labeled', 'w')

    amount = 500

    for seq in create_pos_examples(amount):
        pos.write("{0}\n".format(seq))    
    pos.close()

    for seq in create_neg_examples(amount, MAX_DIGITS):
        neg.write("{0}\n".format(seq))    
    neg.close()

    for inputs in create_data_with_labels(1000, MAX_DIGITS):
        train.write("{0}\n".format(inputs))
    train.close()

    for inputs in create_data_with_labels(500, MAX_DIGITS):
        dev.write("{0}\n".format(inputs))
    dev.close()

    not_labeled, labeled = create_test_files(200, MAX_DIGITS)
    for inputs in not_labeled:
        test.write("{0}\n".format(inputs))
    test.close()
    for inputs in labeled:
        test_labeled.write("{0}\n".format(inputs))
    test_labeled.close()
    

if __name__ == "__main__":
    main()