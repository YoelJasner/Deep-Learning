STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}
         
import dynet as dy
import numpy as np

UNKNOWN = "<UNK>"

# implemenation based on the slides of Dr. Yoav Goldberg
# http://sag.art.uniroma2.it/clic2017/clic-2017_goldberg_dynet_tutorial.pdf
# just a note for me so I can find a good tutorial

class BiLSTM:
    def __init__(self, pc, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim):
        self.pc = pc
        
        # first layer of biLSTM
        self.fwdRNN1 = dy.LSTMBuilder(lstm_layers, repr_dim, lstm_dim1, self.pc)
        self.bwdRNN1 = dy.LSTMBuilder(lstm_layers, repr_dim, lstm_dim1, self.pc)
        
        # second layer of biLSTM
        self.fwdRNN2 = dy.LSTMBuilder(lstm_layers, 2 * lstm_dim1, lstm_dim2, self.pc)      # input layer is multiplied by two because we concat
        self.bwdRNN2 = dy.LSTMBuilder(lstm_layers, 2 * lstm_dim1, lstm_dim2, self.pc)      # the outputs of the biLSTM of the previous layer
        
        self.W = self.pc.add_parameters((out_dim, 2 * lstm_dim2))                          # multiplied by two as before
        self.b = self.pc.add_parameters(out_dim)

    def __call__(self, words):
        """
        Inputs sentence to the network: Get representation of sentence, which is fed to the first
        biLSTM, getting b_1 to b_n. Then inserted into the second biLSTM, thus getting b'_1 up
        to b'_n. Then each b'_i is fed to the MLP1 and the the output returns after a softmax is
        applied.

        :param sentence: Input sentence.
        :return: Softmax vector of the output vector of the net.
        """
        dy.renew_cg()        

        f_init = self.fwdRNN1.initial_state()
        b_init = self.bwdRNN1.initial_state()
        wembs = [self.repr(w) for w in words]
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(wembs[::-1])         # reversing the inputs for backward lstm
        
        bi = [dy.concatenate([f, b]) for f,b in zip(fw_exps, bw_exps[::-1])]

        f_init = self.fwdRNN2.initial_state()
        b_init = self.bwdRNN2.initial_state()
        fw_exps = f_init.transduce(bi)
        bw_exps = b_init.transduce(bi[::-1])         # reversing the inputs for backward lstm

        bi = [dy.concatenate([f, b]) for f,b in zip(fw_exps, bw_exps[::-1])]  

        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        return [dy.softmax(W * x + b) for x in bi]

    def create_network_pred_loss(self, inputs, expected_answer):
        probs = self(inputs)
        tags = [np.argmax(prb.npvalue()) for prb in probs]
        loss =  [-dy.log(dy.pick(prb, exp)) for prb,exp in zip(probs, expected_answer)]
        return tags, loss
    
    def create_network_return_prediction(self, inputs):
        probs = self(inputs)
        return [np.argmax(prb.npvalue()) for prb in probs]

    def save_to(self, file_name):
        self.pc.save(file_name)

    def load_from(self, file_name):
        self.pc.populate(file_name)

    def repr(self, word):
        pass


class WordEmbeddingNet(BiLSTM):
    """
    option (a):
    the word is embedded to a vector using an embedding matrix
    """
    def __init__(self, pc, X2I, vocab_size, embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim):
        BiLSTM.__init__(self, pc, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim)
        self.X2I = X2I
        # word embedding
        self.E = self.pc.add_lookup_parameters((vocab_size, embed_dim))

    def repr(self, word):
        word = word if word in self.X2I else UNKNOWN
        return dy.lookup(self.E, self.X2I[word])

class CharEmbeddedLSTMNet(BiLSTM):
    """
    option (b):
    each letter of the word is embedded to a vector using the embedding matrix and then inputted into a
    LSTM, whose output is the word vector representation.
    """
    def __init__(self, pc, X2I, vocab_size, embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim):
        BiLSTM.__init__(self, pc, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim)
        self.X2I = X2I
        # letter embedding
        self.E = pc.add_lookup_parameters((vocab_size, embed_dim))
        # lstm layer
        self.lstm = dy.LSTMBuilder(1, embed_dim, repr_dim, pc)
    
    def repr(self, word):
        state = self.lstm.initial_state()
        x_t = [self.E[self.X2I[c]] if c in self.X2I else self.E[self.X2I[UNKNOWN]] for c in word]
        outputs = state.transduce(x_t)
        return outputs[-1]

class SubwordEmbeddedNet(BiLSTM):
    '''
    option (c)
    every word, its prefix and suffix get en embedding layer, the sum of them represent the word
    '''
    def __init__(self, pc, X2I, vocab_size, embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim):
        BiLSTM.__init__(self, pc, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim)
        self.X2I = X2I
        # word embedding
        self.E = self.pc.add_lookup_parameters((vocab_size, embed_dim))

    def split_word(self, word):
        if len(word) <= 3:
            return [word]
        
        return [word[:3], word, word[-3:]]
    
    def repr(self, word):
        word = word if word in self.X2I else UNKNOWN
        return dy.esum([self.E[self.X2I[p]] if p in self.X2I else self.E[self.X2I[UNKNOWN]] for p in self.split_word(word)])

class WordCharEmbeddedNet(BiLSTM):
    '''
    option (d)
    first the word is embedded to a vector using an embedding matrix
    next each letter of the word is embedded to a vector using the embedding matrix and then inputted into a
    LSTM
    the two representations we get are concatenated and inserted a linear layer
    '''
    def __init__(self, pc, X2I, vocab_size, embed_dim, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim):
        BiLSTM.__init__(self, pc, lstm_layers, repr_dim, lstm_dim1, lstm_dim2, out_dim)
        self.X2I = X2I
        # word and character embedding
        self.E = pc.add_lookup_parameters((vocab_size, embed_dim))
        # lstm layer
        self.lstm = dy.LSTMBuilder(1, embed_dim, repr_dim, pc)

        self.W0 = self.pc.add_parameters((repr_dim, 2*repr_dim))
        self.b0 = self.pc.add_parameters(repr_dim)

    def repr(self, word):
        wemb = self.E[self.X2I[word]] if word in self.X2I else self.E[self.X2I[UNKNOWN]]

        state = self.lstm.initial_state()
        x_t = [self.E[self.X2I[c]] if c in self.X2I else self.E[self.X2I[UNKNOWN]] for c in word]
        cemb = state.transduce(x_t)[-1]

        out = dy.concatenate([wemb, cemb])
        return self.W0*out + self.b0
