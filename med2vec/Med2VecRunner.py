import tensorflow as tf
import numpy as np
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


# create two vectors that contain all permutations of codes of a visit
def pickTwo(codes, iVector, jVector):
    for first in codes:
        for second in codes:
            if first == second:
                continue
            iVector.append(first)
            jVector.append(second)


# written by me
def pad_matrix_for_prediction(seqs, labels, config):
    """
    For prediction where we use the same file for sequence and labels,
    we don't want the check on n_input > n_output

    n_input will equal n_output when we're using complete ICD9 codes,
    but we still need to return y (codes) as targets for prediction

    :param seqs: list of lists
    :param labels: list of lists
    :param config: dictionary
    :return: multi-hot encoded, padded representation of x (sequence), y (labels), mask,
      and 2 vectors containing every permutation of a code in a sequence with every other code in the sequence
    """
    n_samples = len(seqs)
    i_vec = []
    j_vec = []
    n_input = config['n_input']
    n_output = config['n_output']

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


# return multi-hot encoded representation of (and mask for) medical codes
# also return all permutations of possible pairs of codes per visit (ivec, jvec)
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


# written by me
def get_med2vec_to_id_dict(resource_path):
    # return dictionary of the mapping of Med2Vec complete ICD9 codes (sequences) to Med2Vec IDs
    with open(resource_path + 'processed.types', 'r') as f1:
        for line in f1:
            med2vec_code_to_id_seq = eval(line)

    return med2vec_code_to_id_seq


# written by me
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return 'Key doesn\'t exist'


# written by me
def get_care_condition_for_med2vec_id(code, resource_path):
    # get Med2Vec ICD9 code for Med2Vec ID
    med2vec_code_to_id_seq = get_med2vec_to_id_dict(resource_path)
    med2vec_key = get_key(med2vec_code_to_id_seq, code)  # D_728.86

    # get MIMIC-III ICD9 code for Med2Vec ICD9 code
    med2vec_to_mimic3_seqs = pickle.load(open(resource_path + 'seqs_to_icd.dict', 'rb'))
    mimic3_code = med2vec_to_mimic3_seqs[med2vec_key]  # 72886

    # does this code belong to one of the 25 care conditions from the Benchmark task
    task_nr_to_phen_codes = pickle.load(open(resource_path + 'task_nr_to_phenotype_codes.dict', 'rb'))
    for task_nr, phen_codes in task_nr_to_phen_codes.items():
        if mimic3_code in phen_codes:
            return task_nr

    return None


# written by me
# fill care_condition vector with their (highest) predicted probabilities
def predict_conditions(y_true, y_pred, config):
    # care prediction matrix:
    # 1 row per patient
    # 25 columns: 1 for each care condition
    # each element should have the -highest- predicted probability
    # for ANY of the predicted codes that map to that care condition
    care_preds = np.zeros([len(y_true), 25], dtype=float)
    care_true = np.zeros([len(y_true), 25], dtype=float)

    resource_path = 'resources/'

    for pat_idx, last_visit in enumerate(y_true):
        for idx, code in enumerate(last_visit):
            if code == 1.:
                task_nr = get_care_condition_for_med2vec_id(idx, resource_path)

                if task_nr is not None:
                    task_index = task_nr - 1
                    care_true[pat_idx][task_index] = 1.

                    probit = y_pred[pat_idx][idx]

                    # if the care condition idx already has a probability
                    # replace it with this one if this one is higher
                    if probit > care_preds[pat_idx][task_index]:
                        care_preds[pat_idx][task_index] = probit

    # write to file so we can analyze them later
    model_path = config['model_path']
    with open(model_path + '/care_true.pkl', 'wb') as f1:
        pickle.dump(care_true, f1)

    with open(model_path + '/care_preds.pkl', 'wb') as f1:
        pickle.dump(care_preds, f1)


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


# altered to ignore non-sequence patient separators
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

            # demographics file, codes grouped for labels
            if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]

                # don't calculate cost on non-sequence
                if len(x_batch[0]) == 1 and sum(x_batch[0]) == -1:
                    print('separator: skipped')
                    continue

                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)

            # demographics file, codes not grouped
            elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]

                # don't calculate cost on non-sequence
                if len(x_batch[0]) == 1 and sum(x_batch[0]) == -1:
                    print('separator: skipped')
                    continue

                x, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, mask=mask, i_vec=i_vec, j_vec=j_vec)

            # no demographics file, codes grouped for labels
            elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]

                # don't calculate cost on non-sequence
                if len(x_batch[0]) == 1 and sum(x_batch[0]) == -1:
                    print('separator: skipped')
                    continue

                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)

            # no demographics file, codes not grouped
            else:
                # don't calculate cost on non-sequence
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

        # save trained weights to file
        if epoch == config['max_epoch'] - 1:
            save_path = config['model_path'] + '/med2vec'
            saver.save(
                sess=med2vec.sess,
                save_path=save_path,
                global_step=config['max_epoch']
            )


# re-written completely by me
def predict_next_visit(med2vec, saver, config, top_n=5):
    # get trained model (weights)
    ckpt = tf.train.get_checkpoint_state(config['model_path'])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)

    # open TESTING data
    test_seq_file = 'test_seqs.pkl'
    test_demo_file = ''
    # for complete ICD9 codes
    test_label_file = test_demo_file

    # for 3-digit ICD9 codes
    #test_label_file = 'test_labels.pkl'

    x_seq, d_seq, y_seq = load_data(
        test_seq_file,
        test_demo_file,
        test_label_file
    )

    # get visit representation, input and output for testing data
    # demographics doesn't seem to be represented here
    if config['n_input'] > config['n_output']:
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
    # where visit i (codes of next-to-last visit for a patient) are the input (x_seq_new)
    # and visit j (codes of last visit for a patient) are the target (y_seq_new)
    # visit representation for next-to-last visit (visit i) should be aligned with these two sequences
    x_seq_new = []
    y_seq_new = []
    visit_seq_new = []

    # append a [-1] to the end of x_seq so we also get the very last visit at the end of the sequence file
    x_seq = np.append(x_seq, np.array([-1]), axis=0)

    for x_idx in range(len(x_seq)):
        # don't know why the final appended numpy ARRAY of [-1] only ever shows up as -1... whatever
        if x_seq[x_idx] == [-1] or x_seq[x_idx] == -1:
            # take the visit before the last visit as input: this will be next-to-last visit for patient
            x_seq_new.append(x_seq[x_idx - 2])

            # take last visit (right before this x_idx) as target
            y_seq_new.append(x_seq[x_idx - 1])

            # we want the visit representation for the next-to-last visit
            visit_seq_new.append(visit_seq[x_idx - 2])

    # fit model with visit representation to predict target codes
    predict_model2 = PredictModel(n_input=config['n_emb'], n_output=config['n_output'])

    total_batch = int(np.ceil(len(x_seq_new) / config['batch_size']))

    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            visit_batch = visit_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]

            # use pad_matrix_for_prediction(), since when we predict,
            # our target labels are not identical to our input sequences anymore
            x, y, mask, i_vec, j_vec = pad_matrix_for_prediction(x_batch, y_batch, config)

            # minimize cost using visit representation and target labels
            cost = predict_model2.partial_fit(x=visit_batch, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # get padded representation of true codes for last visit (y)
    x, y, mask, i_vec, j_vec = pad_matrix_for_prediction(
        x_seq_new,
        y_seq_new,
        config
    )

    # get predicted probabilities for each possible code using visit representation
    pred_y = predict_model2.get_result(x=visit_seq_new)

    predict_conditions(y, pred_y, config)
    print(precision_top(y, pred_y, rank=top_n))


# altered to use parameters passed in over the command line (data files and model file paths)
def get_config(args):
    config = dict()
    config['init_scale'] = 0.01
    config['n_windows'] = 1
    config['n_input'] = 4894  # 49 for 2 pat test
    config['n_emb'] = 200
    config['n_demo'] = 0
    config['n_hidden'] = 200
    config['n_output'] = 4894  # 49 for 2 pat test, 942 when using train_labels.pkl instead of train_seqs.pkl as label_file
    config['max_epoch'] = 20
    config['n_samples'] = 22138  # train_seqs.pkl, 8 when using 2 pat test
    config['batch_size'] = 256
    config['display_step'] = 1
    config['seq_file'] = args.seq_file
    config['label_file'] = args.label_file
    config['demo_file'] = args.demo_file
    config['model_path'] = args.model_path

    return config


# allow user to pass in different data files and paths for saving models
def parse_arguments(parser):
    parser.add_argument('--seq_file', type=str, default='seqs.pkl', help='The path to the Pickled file containing visit information of patients')
    parser.add_argument('--label_file', type=str, default='labels.pkl', help='The path to the Pickled file containing grouped visit information of patients.')
    parser.add_argument('--demo_file', type=str, default='', help='The path to the Pickled file containing demographic information of patients. If you are not using patient demographic information, do not use this option')
    parser.add_argument('--model_path', type=str, default='./Med2Vec_model/200emb_200hidden', help='The path to the directory where the model with these params should be saved.')

    args = parser.parse_args()
    return args


def main(_):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    config = get_config(args)

    # call this to train and evaluate on a two-patient sample
    # python3 Med2VecRunner.py --seq_file=test_2pat_data/seqs.pkl --label_file=test_2pat_data/seqs.pkl --model_path=./Med2Vec_model/test_2pat_icd_complete

    # call this to train on train data and evaluate on test data, using 3-digit ICD9 codes as labels
    # python3 Med2VecRunner.py --seq_file=train_seqs.pkl --label_file=train_labels.pkl --model_path=./Med2Vec_model/train_test_split

    # call this to train on train data and evaluate on test data, using COMPLETE ICD9 codes as labels
    # python3 Med2VecRunner.py --seq_file=train_seqs.pkl --label_file=train_seqs.pkl --model_path=./Med2Vec_model/icd_complete_train_test_split

    med2vec = Med2Vec(
        n_input=config['n_input'],
        n_emb=config['n_emb'],
        n_demo=config['n_demo'],
        n_hidden=config['n_hidden'],
        n_output=config['n_output'],
        n_windows=config['n_windows']
    )

    saver = tf.train.Saver()
    #model_train(med2vec, saver, config)
    #predict_next_visit(med2vec, saver, config, top_n=5)


if __name__ == "__main__":
    tf.app.run()
