import numpy as np
import pickle


def load_data(x_file, d_file, y_file):
    x_file = '../Med2Vec_data/' + x_file
    x_seq = np.array(pickle.load(open(x_file, 'rb')), dtype='object')

    d_seq = []
    if len(d_file) > 0:
        d_file = '../Med2Vec_data/' + d_file
        d_seq = np.array(pickle.load(open(d_file, 'rb')), dtype='object')

    y_seq = []
    if len(y_file) > 0:
        y_file = '../Med2Vec_data/' + y_file
        y_seq = np.array(pickle.load(open(y_file, 'rb')), dtype='object')

    return x_seq, d_seq, y_seq


train_seq, d_seq, test_seq = load_data(
    'train_seqs.pkl',
    '',
    'test_seqs.pkl'
)

print('len(train_seq)', len(train_seq))
print('len(test_seq)', len(test_seq))
