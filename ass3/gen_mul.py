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
        a#b#c where a*b=c
        param amount: size of array
        param max_digits: maximum amount of digits in a sequence
    '''
    data = []
    for _ in range(amount):
        a = r.randint(1, 10**MAX_DIGITS)
        b = r.randint(1, 10**MAX_DIGITS)
        c = a * b
        data.append("{0}#{1}#{2}".format(a,b,c))
    return data

def create_neg_examples(amount, max_digits=MAX_DIGITS):
    '''
        returns an array of sequences in form of:
        a#b#c where a*b!=c
        param amount: size of array
        param max_digits: maximum amount of digits in a sequence
    '''
    data = []
    for _ in range(amount):
        a = r.randint(1, 10**MAX_DIGITS)
        b = r.randint(1, 10**MAX_DIGITS)
        c = a * b - 1
        data.append("{0}#{1}#{2}".format(a,b,c))
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
    language = 'mul_'
    train = open(language + 'train', 'w')
    dev = open(language + 'dev', 'w')
    test = open(language + 'test', 'w')
    test_labeled = open(language + 'test_labeled', 'w')

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