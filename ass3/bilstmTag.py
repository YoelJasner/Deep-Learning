STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}

from sys import argv
import bilstm_model as bilstm
import json
import yaml
import dynet as dy

L2I = {}
I2L = {}
X2I = {}
UNKNOWN = "<UNK>"


def get_data(filename):
    data = []
    sentence = []
    for line in open(filename):
        if line == '\n':
            data.append(sentence)
            sentence = []
            continue
        sentence.append(line.strip())
    return data

def predict(net, inputs, filename):
    '''
    calculating predictions on 'inputs' and writing to 'filename'
    '''
    with open(filename, "w") as f:
        for sentence in inputs:
            y_hat = net.create_network_return_prediction(sentence)
            label = [I2L[l] for l in y_hat]
            for w,l in zip(sentence, label):
                f.write('{0} {1}\n'.format(w, l))
            f.write('\n')

def main():
    if len(argv) != 6:
        print("Wrong command-line arguments!")
        exit()

    global X2I,L2I,I2L

    # parameters passed by the command-line
    repr_c = argv[1]
    modelfile = argv[2]
    inputfile = argv[3]
    parameterfile = argv[4]
    outputfile = argv[5]

    with open(parameterfile, "r") as read_file:
        parameters = yaml.safe_load(read_file)

    # loading sizes for network components
    embed_dim = parameters["embed_dim"]
    repr_dim = parameters["repr_dim"]
    lstm_layers = parameters["lstm_layers"]
    lstm_dim1 = parameters["lstm_dim1"]
    lstm_dim2 = parameters["lstm_dim2"]

    WORDS = parameters["WORDS"]
    L2I = parameters["L2I"]
    I2L = {i:l for l,i in L2I.items()}
    X2I = {}

    pc = dy.ParameterCollection()
    
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
    
    # loading a model
    net.load_from(modelfile)

    TEST = get_data(inputfile)

    predict(net, TEST, outputfile)


if __name__ == "__main__":
    main()