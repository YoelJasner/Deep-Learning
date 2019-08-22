STUDENT={'name': 'STEVE GUTFREUND_YOEL JASNER',
         'ID': '342873791_204380992'}
         
import numpy as np

VOCAB = []
for l in open('vocab.txt'):
    VOCAB.append(l.strip())

V2I = {w:i for i,w in enumerate(VOCAB)}
VOCAB = np.array(VOCAB)
EMBED = np.loadtxt('wordVectors.txt', dtype=float)

N_EMBED = EMBED.copy()

norm = np.linalg.norm(N_EMBED, axis=1)
for i,line in enumerate(N_EMBED):
    N_EMBED[i] = np.true_divide(line, norm[i])

def most_similar(word, k):
    v_word = N_EMBED[V2I[word]]
    dist_vector = np.abs(N_EMBED.dot(v_word) - v_word.dot(v_word))
    idx = np.argpartition(dist_vector, k+1)
    index = np.argwhere(idx==V2I[word])
    idx = np.delete(idx, index)
    return VOCAB[idx[:k]]

if __name__ == "__main__":
    k = 5
    words = ["dog", "england", "john", "explode", "office"]
    for word in words:
        print(most_similar(word, k))