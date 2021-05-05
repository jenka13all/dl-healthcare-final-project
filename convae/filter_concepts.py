import os.path
from csv import reader, DictReader
from collections import defaultdict
import pickle


# returns the total number of patients ("sentences")
def get_doc_length(train_file, test_file):
    train_file.seek(0)
    test_file.seek(0)
    doc_length = 0

    # skip header
    train_file.readline()
    for line in train_file:
        doc_length += 1

    # skip header
    test_file.readline()
    for line in test_file:
        doc_length += 1

    return doc_length


# returns the number of patients ("sentences") with at least one instance of the item_id ("word" from vocabulary)
def get_item_count(train_file, test_file, item_id):
    train_file.seek(0)
    test_file.seek(0)
    nr_sequences = 0

    train_reader = reader(train_file)
    for row in train_reader:
        if item_id in row:
            nr_sequences += 1

    test_reader = reader(test_file)
    for row in test_reader:
        if item_id in row:
            nr_sequences += 1

    return nr_sequences


# sums the number of times an item appears in a patient sequence, divided by that sequence's length
def get_item_proportion_per_seq(train_file, test_file, item_id):
    train_file.seek(0)
    test_file.seek(0)
    tf = 0

    # skip header
    train_file.readline()
    for line in train_file:
        # split line by comma
        seq_items = line.strip().split(',')
        seq_length = len(seq_items)
        seq_item_count = seq_items.count(item_id)
        tf += (float(seq_item_count) / float(seq_length))

    # skip header
    test_file.readline()
    for line in test_file:
        # split line by comma
        seq_items = line.strip().split(',')
        seq_length = len(seq_items)
        seq_item_count = seq_items.count(item_id)
        tf += (float(seq_item_count) / float(seq_length))

    # don't round
    return tf


# returns the frequency of each vocab item ("word") in patient sequences
# in proportion to the total number of all patient sequences ("document")
def get_doc_freq(train_seq, test_seq, dict_file):
    if os.path.isfile(dict_file):
        return pickle.load(open(dict_file, 'rb'))

    doc_length = get_doc_length(train_seq, test_seq)

    doc_freq = defaultdict(float)
    with open('data/cohort-vocab.csv', 'r') as vocab:
        vocab_reader = DictReader(vocab)
        for row in vocab_reader:
            item = str(row['INDEX'])
            item_count = get_item_count(train_seq, test_seq, item)
            # don't round
            doc_freq[item] = float(item_count)/float(doc_length)

    with open(dict_file, 'wb') as f1:
        pickle.dump(doc_freq, f1)

    return doc_freq


# returns the sum of the ratios of item frequency in a patient sequence
# to the length of that patient sequence
# for each item over the whole set of patient sequences
def get_term_freq(train_seq, test_seq, dict_file):
    if os.path.isfile(dict_file):
        return pickle.load(open(dict_file, 'rb'))

    term_freq = defaultdict(float)
    with open('data/cohort-vocab.csv', 'r') as vocab:
        vocab_reader = DictReader(vocab)
        for row in vocab_reader:
            item = str(row['INDEX'])
            tf = get_item_proportion_per_seq(train_seq, test_seq, item)
            term_freq[item] = tf

    with open(dict_file, 'wb') as f1:
        pickle.dump(term_freq, f1)

    return term_freq


def make_filter_scores_dict(df_dict, tf_dict, dict_file):
    if os.path.isfile(dict_file):
        return pickle.load(open(dict_file, 'rb'))

    filter_scores = {key: tf_dict[key] * df_dict.get(key, 0) for key in tf_dict.keys()}

    with open(dict_file, 'wb') as f1:
        pickle.dump(filter_scores, f1)

    return filter_scores


train_seq = open('data/cohort-ehrseq.csv', 'r')
test_seq = open('data/cohort_test-ehrseq.csv', 'r')

doc_freq_dict = get_doc_freq(train_seq, test_seq, 'resources/doc_freq_dict.pkl')
term_freq_dict = get_term_freq(train_seq, test_seq, 'resources/term_freq_dict.pkl')
filter_scores_dict = make_filter_scores_dict(doc_freq_dict, term_freq_dict, 'resources/filter_scores_dict.pkl')

train_seq.close()
test_seq.close()

#print(len(filter_scores_dict))

# number of terms with score of < 10e-6
#res = sum(x < 0.000001 for x in filter_scores_dict.values())
#print('nr of scores = 0', res)

# what's the distribution of filter scores?
#import matplotlib.pyplot as plt
#plt.hist(filter_scores.values(), color='blue', edgecolor='black', bins=200)
#plt.savefig('figures/filter_scores_distribution.png', dpi=700)
