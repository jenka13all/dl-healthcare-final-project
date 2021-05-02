import numpy as np
import pickle


def load_data(x_file, d_file, y_file):
    x_file = './Med2Vec_data/' + x_file
    x_seq = np.array(pickle.load(open(x_file, 'rb')), dtype='object')

    d_seq = []
    if len(d_file) > 0:
        d_file = './Med2Vec_data/' + d_file
        d_seq = np.array(pickle.load(open(d_file, 'rb')), dtype='object')

    y_seq = []
    if len(y_file) > 0:
        y_file = './Med2Vec_data/' + y_file
        y_seq = np.array(pickle.load(open(y_file, 'rb')), dtype='object')

    return x_seq, d_seq, y_seq


x_seq, d_seq, y_seq = load_data(
    'seqs.pkl',
    '',
    'labels.pkl'
)

splittable_indexes = []
for index, seq in enumerate(x_seq):
    if seq == [-1]:
        splittable_indexes.append(index)

print('splits in x_seq', len(splittable_indexes))  # 7536
print('80% of x_seq', int(.8 * len(splittable_indexes)))  # 6028
train_end_index = splittable_indexes[6028]
training_seq = x_seq[0:train_end_index]
testing_seq = x_seq[train_end_index+1:]

with open('Med2Vec_data/train_seqs.pkl', 'wb') as f1:
    pickle.dump(training_seq, f1)

with open('Med2Vec_data/test_seqs.pkl', 'wb') as f1:
    pickle.dump(testing_seq, f1)

label_indexes = []
for index, seq in enumerate(y_seq):
    if seq == [-1]:
        label_indexes.append(index)

print('splits in y_seq', len(label_indexes))  # 7536
print('80% of y_seq', int(.8 * len(label_indexes)))  # 6028
train_end_index = label_indexes[6028]
training_labels = y_seq[0:train_end_index]
testing_labels = y_seq[train_end_index+1:]

with open('Med2Vec_data/train_labels.pkl', 'wb') as f1:
    pickle.dump(training_labels, f1)

with open('Med2Vec_data/test_labels.pkl', 'wb') as f1:
    pickle.dump(testing_labels, f1)
