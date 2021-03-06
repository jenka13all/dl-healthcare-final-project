from model.data_loader import EHRdata, ehr_collate
from gensim.models import Word2Vec
from train import train_and_evaluate
from time import time
from torch.utils.data import DataLoader
import model.net as net
import torch.nn as nn
import torch
import utils as ut
import vocabulary
import argparse
import sys
import csv
import os

"""
Learn patient representations from the EHRs using an autoencoder of CNNs
"""


def write_ehr_subseq(data_generator, outfile):
    ehr_subseq = {}
    for list_m, batch in data_generator:
        for b, m in zip(batch, list_m):
            if len(b) == 1:
                ehr_subseq[m] = b.tolist()
            else:
                seq = []
                for vec in b.tolist():
                    seq.extend(vec)
                nseq, nleft = divmod(len(seq), ut.len_padded)
                if nleft > 0:
                    seq = seq + [0] * (ut.len_padded - nleft)

                for i in range(0, len(seq) - ut.len_padded + 1, ut.len_padded):
                    ehr_subseq.setdefault(m, list()).append(seq[i:i + ut.len_padded])

    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "EHRsubseq"])
        for m, subseq in ehr_subseq.items():
            for seq in subseq:
                wr.writerow([m] + list(filter(lambda x: x != 0, seq)))


def learn_patient_representations(
        indir,
        test_set=False,
        sampling=None,
        emb_filename=None
):
    # encodings folder to save the representations
    exp_dir = os.path.join(indir, 'encodings')
    if test_set:
        exp_dir = os.path.join(indir, 'encodings', 'test')

    os.makedirs(exp_dir, exist_ok=True)

    # get the vocabulary size
    vocab_size, vocab = vocabulary.get_vocab(indir)

    # load pre-computed embeddings
    if emb_filename is not None:
        model = Word2Vec.load(emb_filename)
        embs = model.wv
        del model
        print('Loaded pre-computed embeddings for {0} concepts'.format(
            len(embs.vocab)))
    else:
        embs = None

    # set random seed for experiment reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # load data
    data_tr = EHRdata(os.path.join(indir), ut.dt_files['ehr-file'], sampling)
    data_generator_tr = DataLoader(
        data_tr,
        ut.model_param['batch_size'],
        shuffle=True,
        collate_fn=ehr_collate
    )

    if test_set:
        data_ts = EHRdata(os.path.join(indir), ut.dt_files['ehr-file-test'], sampling)

        data_generator_ts = DataLoader(
            data_ts,
            ut.model_param['batch_size'],
            shuffle=True,
            collate_fn=ehr_collate
        )
        print("Test cohort size: {0}".format(len(data_ts)))
    else:
        data_generator_ts = data_generator_tr

    print('Training cohort size: {0}\n'.format(len(data_tr)))
    print('Max Sequence Length: {0}\n'.format(ut.len_padded))
    print('Learning rate: {0}'.format(ut.model_param['learning_rate']))
    print('Batch size: {0}'.format(ut.model_param['batch_size']))
    print('Kernel size: {0}\n'.format(ut.model_param['kernel_size']))

    # define model and optimizer
    model = net.ehrEncoding(
        vocab_size=vocab_size,
        max_seq_len=ut.len_padded,  # 32
        emb_size=ut.model_param['embedding_size'],  # 100
        kernel_size=ut.model_param['kernel_size'],  # 5
        pre_embs=embs,
        vocab=vocab
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ut.model_param['learning_rate'],
        weight_decay=ut.model_param['weight_decay']
    )

    # model.cuda()
    if torch.cuda.device_count() > 1:
        print('No. of GPUs: {0}\n'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        model.cuda()
        print('No. of GPUs: 1\n')

    loss_fn = net.criterion
    print('Training for {} epochs\n'.format(ut.model_param['num_epochs']))

    #only train
    train_and_evaluate(
        model,
        data_generator_tr,
        data_generator_ts,
        loss_fn,
        optimizer,
        net.metrics,
        exp_dir
    )

    # uncomment this out to train AND evaluate
    # will take a really, really long time
    # training and evaluation
    # results of best model are saved to outdir/best_model.pt in this function
    '''
    mrn, encoded, encoded_avg, metrics_avg = train_and_evaluate(
        model,
        data_generator_tr,
        data_generator_ts,
        loss_fn,
        optimizer,
        net.metrics,
        exp_dir
    )

    # save encodings
    # encoded vectors (representations)
    outfile = os.path.join(exp_dir, 'convae_avg_vect.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "ENCODED-AVG"])
        for m, e in zip(mrn, encoded_avg):
            wr.writerow([m] + list(e))

    outfile = os.path.join(exp_dir, 'convae_vect.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "ENCODED-SUBSEQ"])
        for m, evs in zip(mrn, encoded):
            for e in evs:
                wr.writerow([m] + e)

    # metrics (loss and accuracy)
    outfile = os.path.join(exp_dir, 'metrics.txt')
    with open(outfile, 'w') as f:
        f.write('Mean loss: %.3f\n' % metrics_avg['loss'])
        f.write('Accuracy: %.3f\n' % metrics_avg['accuracy'])

    # chop patient sequences into fixed subsequences of length L
    # L = ut.len_padded = 32
    # I think that this is here for the human-readable version of how the patient records are subset
    outfile = os.path.join(exp_dir, 'cohort_ehr_subseq{0}.csv'.format(ut.len_padded))
    write_ehr_subseq(data_generator_tr, outfile)

    if test_set:
        outfile = os.path.join(exp_dir, 'test_cohort_ehr_subseq{0}.csv'.format(ut.len_padded))
        write_ehr_subseq(data_generator_ts, outfile)
    '''
    return


# main function

def _process_args():
    parser = argparse.ArgumentParser(
        description='EHR Patient Stratification: derive patient representations from longitudinal EHRs')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument('--test_set', dest='test_set', default=False)
    parser.add_argument('-s', default=None, type=int, help='Enable sub-sampling with data size (default: None)')
    parser.add_argument('-e', default=None, help='Pre-computed embeddings (default: None)')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('indir', args.indir)
    print('test_set', args.test_set)

    start = time()
    learn_patient_representations(
        indir=args.indir,
        test_set=args.test_set,
        sampling=args.s,
        emb_filename=args.e
    )

    print('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print('Task completed\n')
