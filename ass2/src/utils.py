STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}
         
import numpy as np

LABELS = []
TRAIN = []
DEV = []
TEST = []
EMBED = []
L2I = {}
I2L = {}
W2I = {}
global TASK

def split_word(word):
    if len(word) <= 3:
        return [word]
    
    return [word[:3], word, word[-3:]]

def create_5_window(sentence):
    '''
    given a sentence, it returns list of windows of size 5
    '''
    START = '*START*'
    END = '*END*'
    windows = []
    sentence = [START, START] + sentence + [END, END]
    for i in range(2, len(sentence) - 2):
        windows.append([sentence[i-2], sentence[i-1], sentence[i], sentence[i+1], sentence[i+2]])
    return windows

def get_data(filename, task, update_vocab=True, pre_trained=False, sub_words=False):
    '''
    returns the data stored in 'filename', where each word get's its label based on the 2 words before it and 2 after it
    so each example contains 5 words and a label for the middle one
    W2I - dictionarry of vocabulary of all words in training set
    LABELS - list of all possible lables
    '''
    print("Loading {0} data...".format(task))

    global LABELS

    inputs = []
    labels = []
    sentence = []
    for l in open(filename):
        if l == '\n':                           # end of a sentence
            inputs.extend(create_5_window(sentence))
            sentence = []
            continue
        x,y = l.strip().split()
        if pre_trained:                         # for part 3, lower casing everything
            x = x.lower()
        if update_vocab:                        # unlike for DEV set
            LABELS.append(y)                    # adding the label to list of all labels
            if x not in W2I:
                if sub_words:                   # if we are interested in sub_words embedding
                    for part in split_word(x):
                        if part not in W2I:
                            W2I[part] = len(W2I)
                else:
                    W2I[x] = len(W2I)           # adding the word to dictionary
        sentence.append(x)
        labels.append(y)
            
    if update_vocab:
        W2I['*START*'] = len(W2I)               # word added at the beginning of each sentence
        W2I['*END*'] = len(W2I)                 # word added at the end of each sentence
        W2I['*UNKNOWN*'] = len(W2I)             # for all unkonwn words
        LABELS = list(set(LABELS))              # removing all duplicates
    
    print("Done.")
    return zip(inputs, labels)

def get_test_data(filename, task, pre_trained=False):
    print("Loading {0} data...".format(task))
    inputs = []
    sentence = []
    for l in open(filename):
        if l == '\n':
            inputs.extend(create_5_window(sentence))
            inputs.append('\n')
            sentence = []
            continue
        x = l.strip()
        #if pre_trained:
        #    x = x.lower()
        sentence.append(x)
    print("Done.")
    return inputs

def epsilon(n, m):
    return np.sqrt(6) / np.sqrt(n+m)

def run(task, pre_trained=False, sub_words=False):
    global TRAIN,DEV,TEST,L2I,I2L,W2I,TASK,EMBED

    TASK = task

    if pre_trained:
        for l in open('vocab.txt'):
            W2I[l.strip()] = len(W2I)            
        EMBED = np.loadtxt('wordVectors.txt', dtype=float).tolist()
    
    TRAIN = get_data(TASK + '/train', 'TRAIN', pre_trained=pre_trained, sub_words=sub_words)
    
    if pre_trained:
        # adding layers to embedding matrix for the 'new' words found in training set
        n = len(EMBED[0])
        for _ in range(len(W2I) - len(EMBED)):
            EMBED.append(np.zeros(n).tolist())
            #EMBED.append(np.ones(n).tolist())
            #EMBED.append(np.random.uniform(-epsilon(1, n), epsilon(1, n), [1, n]).tolist()[0])
    
    # label strings to IDs
    L2I = {l:i for i,l in enumerate(LABELS)}
    # IDs to labels
    I2L = {i:l for i,l in enumerate(LABELS)}
    
    DEV = get_data(TASK + '/dev', 'DEV', update_vocab=False, pre_trained=pre_trained)
    TEST = get_test_data(TASK + '/test', 'TEST', pre_trained=pre_trained)