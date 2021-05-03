import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pickle
from Med2Vec import Med2Vec
from PredictModel import PredictModel
import argparse


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


def pickTwo(codes, iVector, jVector):
    for first in codes:
        for second in codes:
            if first == second:
                continue
            iVector.append(first)
            jVector.append(second)


def pad_matrix(seqs, labels, config):
    n_samples = len(seqs)
    i_vec = []
    j_vec = []
    n_input = config['n_input']
    n_output = config['n_output']

    if n_input > n_output:
        x = np.zeros((n_samples, n_input))
        y = np.zeros((n_samples, n_output))
        mask = np.zeros((n_samples, 1))

        for idx, (seq, label) in enumerate(zip(seqs, labels)):
            if not seq[0] == -1:
                x[idx][seq] = 1.
                y[idx][label] = 1.
                pickTwo(seq, i_vec, j_vec)
                mask[idx] = 1.

        return x, y, mask, i_vec, j_vec
    else:
        x = np.zeros((n_samples, n_input))
        mask = np.zeros((n_samples, 1))
        for idx, seq in enumerate(seqs):
            if not seq[0] == -1:
                x[idx][seq] = 1.
                pickTwo(seq, i_vec, j_vec)
                mask[idx] = 1.

        return x, mask, i_vec, j_vec


# for the top n predicted diagnoses, how many overlap with the true diagnoses?
def precision_top(y_true, y_pred, rank=None):
    if rank is None:
        rank = [1, 2, 3, 4, 5]
    else:
        rank = range(1, rank+1)
    pre = list()

    for i in range(len(y_pred)):
        thisOne = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_pred[i])
            for rk in rank:
                if len(
                        set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))
                ) >= 1:
                    thisOne.append(1)
                else:
                    thisOne.append(0)
            pre.append(thisOne)

    return (np.array(pre)).mean(axis=0).tolist()


def model_train(med2vec, saver, config):
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        total_batch = int(np.ceil(config['n_samples'] / config['batch_size']))
        x_seq, d_seq, y_seq = load_data(
            config['seq_file'],
            config['demo_file'],
            config['label_file']
        )
        # Loop over all batches
        for index in range(total_batch):
            print('index', index)
            x_batch = x_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = []
            if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)
            elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                x, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, mask=mask, i_vec=i_vec, j_vec=j_vec)
            elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                if len(x_batch[0]) == 1 and sum(x_batch[0]) == -1:
                    print('separator: skipped')
                    continue
                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)
            else:
                if len(x_batch[0]) == 1 and sum(x_batch[0]) == -1:
                    print('separator: skipped')
                    continue
                x, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, mask=mask, i_vec=i_vec, j_vec=j_vec)
            # Compute average loss
            if np.isnan(cost) or cost < 0:
                cost = 0.
            avg_cost += (cost / config['n_samples']) * config['batch_size']
        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", avg_cost)
        if epoch == config['max_epoch'] - 1:
            save_path = config['model_path'] + '/med2vec'
            saver.save(sess=med2vec.sess,
                       save_path=save_path,
                       global_step=config['max_epoch']
                       )


def show_code_representation(med2vec, saver, config):
    ckpt = tf.train.get_checkpoint_state(config['model_path'])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    w_emb, w_hidden, w_output = med2vec.get_weights()
    labels = ["GXB", "TNB", "GXYB", "NGSHYZ", "MXBDXGY", "NGS", "NCZ"]
    disease_list = [1, 2, 499, 826, 168, 169, 1175]
    plt.scatter(w_emb[disease_list, 0], w_emb[disease_list, 1], s=10)
    for label, x, y in zip(labels, w_emb[disease_list, 0], w_emb[disease_list, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.savefig('./Med2Vec_fig/code_show.png', dpi=700)


def interpret_code_representation(med2vec, saver, config):
    ckpt = tf.train.get_checkpoint_state(config['model_path'])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    w_emb, w_hidden, w_output = med2vec.get_weights()
    code_dict = pickle.load(open("./Med2Vec_data/code_dict.pkl", 'rb'))
    for i in code_dict:
        print(i,code_dict[i])
    for i in range(get_config()['n_emb']):
        print(i,end=' ')
        sorted_code = np.argsort(w_emb[:, i])[get_config()['n_input'] - 10:get_config()['n_input']]
        for j in sorted_code:
            print(code_dict[j], end=' ')
        print()


# I have no idea what they were trying to do here
def predict_next_visit_orig(med2vec, saver, config, top_n=5):
    ckpt = tf.train.get_checkpoint_state('./Med2Vec_model/200emb_200hidden')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    x_seq, d_seq, y_seq = load_data('seqs.pkl', '', 'labels.pkl')
    if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    else:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)

    x_seq, d_seq, y_seq = load_data('seqs.pkl', '', 'labels.pkl')
    x_seq_new = []
    y_seq_new = []
    visit_seq_new = []
    for i in range(config['n_samples'] - 1):
        if x_seq[i][0] != -1 and y_seq[i + 1][0] != -1:
            x_seq_new.append(x_seq[i])
            visit_seq_new.append(visit_seq[i])
            # why are they using the codes of the PREVIOUS visit to x as the target?
            y_seq_new.append(y_seq[i - 1])
    #for i in range(10):
    #    for j in visit_seq_new[i]:
    #        print(j, end=' ')
    #    print()

    predict_model1 = PredictModel(n_input=config['n_input'], n_output=config['n_output'])

    # why are they only using 80% of the "toal" batch size?
    total_batch = int(np.ceil(len(x_seq_new) * 0.8 / config['batch_size']))
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            # how does total_batch fit into this indexing?
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
            cost = predict_model1.partial_fit(x=x, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size'] * 0.8

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    x, y, mask, i_vec, j_vec = pad_matrix(x_seq_new[int(0.8 * len(x_seq_new)):],
                                          y_seq_new[int(0.8 * len(x_seq_new)):], config)
    predict_y = predict_model1.get_result(x=x)
    print(precision_top(y, predict_y, rank=top_n))

    # what purpose do two models have?
    predict_model2 = PredictModel(n_input=config['n_emb'], n_output=config['n_output'])
    total_batch = int(np.ceil(len(x_seq_new) * 0.8 / config['batch_size']))
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            visit_batch = visit_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
            cost = predict_model2.partial_fit(x=visit_batch, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size'] * 0.8

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    # why 80% indexing? is this supposed to be a train/test thing?
    x, y, mask, i_vec, j_vec = pad_matrix(x_seq_new[int(0.8 * len(x_seq_new)):],
                                          y_seq_new[int(0.8 * len(x_seq_new)):], config)

    # visit_seq_new?? and what's up with the indexing again?
    predict_y = predict_model2.get_result(x=visit_seq_new[int(0.8 * len(x_seq_new)):])
    print(precision_top(y, predict_y, rank=top_n))


# so I rewrote it in a way that makes sense to me
def predict_next_visit(med2vec, saver, config, top_n=5):
    # get trained model (weights)
    ckpt = tf.train.get_checkpoint_state(config['model_path'])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)

    # open TESTING data
    test_seq_file = 'test_seqs.pkl'
    test_demo_file = ''
    # for 3-digit ICD9 codes
    #test_label_file = 'test_labels.pkl'
    # for complete ICD9 codes
    test_label_file = 'test_seqs.pkl'


    x_seq, d_seq, y_seq = load_data(
        test_seq_file,
        test_demo_file,
        test_label_file
    )

    # get visit representation, input and output for testing data
    if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    else:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)

    # not sure why we need to do this again, but it can't hurt
    x_seq, d_seq, y_seq = load_data(
        test_seq_file,
        test_demo_file,
        test_label_file
    )

    # create a set of data to predict on, visit i followed by visit j
    # where visit i (codes of visit i) are the input (x_seq_new)
    # and visit j (codes of visit j) are the target (y_seq_new)
    x_seq_new = []
    y_seq_new = []
    visit_seq_new = []
    for i in range(len(x_seq) - 1):
        if x_seq[i][0] != -1 and y_seq[i + 1][0] != -1:
            x_seq_new.append(x_seq[i])
            visit_seq_new.append(visit_seq[i])
            y_seq_new.append(y_seq[i + 1])

    # predict y (from y_seq_new) given x (from x_seq_new)
    predict_model2 = PredictModel(n_input=config['n_emb'], n_output=config['n_output'])

    total_batch = int(np.ceil(len(x_seq_new) / config['batch_size']))

    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            visit_batch = visit_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]

            # process input (x) for getting a visit representation later
            # process target (y) for use in cost minimization, using current visit_batch as the input
            x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
            cost = predict_model2.partial_fit(x=visit_batch, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # get visit representation from learned x
    visit_seq = med2vec.get_visit_representation(x=x)

    # predict target values using visit representation
    predict_y_visit = predict_model2.get_result(x=visit_seq)

    print(precision_top(y, predict_y_visit, rank=top_n))


def get_config(args):
    config = dict()
    config['init_scale'] = 0.01
    config['n_windows'] = 1
    config['n_input'] = 4894
    config['n_emb'] = 200
    config['n_demo'] = 0
    config['n_hidden'] = 200
    config['n_output'] = 4894  # 942 when using train_labels.pkl instead of train_seqs.pkl as label_file
    config['max_epoch'] = 20
    config['n_samples'] = 22138  # train_seqs.pkl
    config['batch_size'] = 256
    config['display_step'] = 1
    config['seq_file'] = args.seq_file
    config['label_file'] = args.label_file
    config['demo_file'] = args.demo_file
    config['model_path'] = args.model_path

    return config


def parse_arguments(parser):
    parser.add_argument('--seq_file', type=str, default='seqs.pkl', help='The path to the Pickled file containing visit information of patients')
    parser.add_argument('--label_file', type=str, default='labels.pkl', help='The path to the Pickled file containing grouped visit information of patients.')
    parser.add_argument('--demo_file', type=str, default='', help='The path to the Pickled file containing demographic information of patients. If you are not using patient demographic information, do not use this option')
    parser.add_argument('--model_path', type=str, default='./Med2Vec_model/200emb_200hidden', help='The path to the directory where the model with these params should be saved.')

    args = parser.parse_args()
    return args


def main(_):
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    config = get_config(args)

    # call this to train on train data and evaluate on test data, using 3-digit ICD9 codes as labels
    # python3 Med2VecRunner.py --seq_file=train_seqs.pkl --label_file=train_labels.pkl --model_path=./Med2Vec_model/train_test_split

    # call this to train on train data and evaluate on test data, using COMPLETE ICD9 codes as labels
    # python3 Med2VecRunner.py --seq_file=train_seqs.pkl --label_file=train_seqs.pkl --model_path=./Med2Vec_model/icd_complete_train_test_split

    med2vec = Med2Vec(n_input=config['n_input'],
                      n_emb=config['n_emb'],
                      n_demo=config['n_demo'],
                      n_hidden=config['n_hidden'],
                      n_output=config['n_output'],
                      n_windows=config['n_windows']
                      )
    saver = tf.train.Saver()
    model_train(med2vec, saver, config)
    #show_code_representation(med2vec, saver, config)
    #predict_next_visit(med2vec, saver, config, top_n=5)
    #interpret_code_representation(med2vec, saver, config)


if __name__ == "__main__":
    tf.app.run()
