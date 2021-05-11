import os
import csv
import utils as ut


def get_vocab(indir):
    # get the vocabulary size
    fvocab = os.path.join(
        os.path.join(indir),
        ut.dt_files['vocab']
    )

    with open(fvocab) as f:
        rd = csv.reader(f)
        next(rd)
        vocab = {}
        for r in rd:
            # index, code
            vocab[int(r[0])] = r[2]
        vocab_size = len(vocab) + 1

    return vocab_size, vocab


# for use when using pre-trained embeddings
# we will want the text descriptions instead of the codes
def get_vocab_descriptive(indir):
    # get the vocabulary size
    fvocab = os.path.join(
        os.path.join(indir),
        ut.dt_files['vocab']
    )

    with open(fvocab) as f:
        rd = csv.reader(f)
        next(rd)
        vocab = {}
        for r in rd:
            # desc, index
            vocab[int(r[3])] = r[0]
        vocab_size = len(vocab) + 1

    return vocab_size, vocab
