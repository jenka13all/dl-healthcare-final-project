import os.path

import numpy as np
from sklearn import metrics
import pickle
import sklearn.utils as sk_utils
import json
import csv


# function from Benchmark study
# https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3models/metrics.py
def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])

    # prevent annoying warning
    if cf[1][1] == 0.0:
        prec1 = 0.0
    else:
        prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}


# function from Benchmark study
# https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3models/metrics.py
def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {
        "auc_scores": auc_scores,
        "ave_auc_micro": ave_auc_micro,
        "ave_auc_macro": ave_auc_macro,
        "ave_auc_weighted": ave_auc_weighted
    }


def get_blank_care_matrix(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        cohort_size = len(list(reader))

    return np.zeros([cohort_size, 25], dtype=float)


def get_care_matrix(care_matrix_filename,
                      pat_data_file,
                      phen_codes_dict,
                      indices_filename):
    if os.path.isfile(care_matrix_filename):
        care_matrix = pickle.load(open(care_matrix_filename, 'rb'))
        indices_to_skip_list = pickle.load(open(indices_filename, 'rb'))

        return care_matrix, indices_to_skip_list

    indices_to_skip = []
    care_matrix = get_blank_care_matrix(pat_data_file)

    with open(pat_data_file) as f:
        rd = csv.reader(f)
        idx = 0
        for r in rd:
            # populate care_matrix
            seqs = r[1:]
            for seq in seqs:
                seq = int(seq)
                if seq in phen_codes_dict:
                    task_nr = int(phen_codes_dict[seq])
                    task_index = task_nr - 1
                    care_matrix[idx][task_index] = 1.

            # populate indices to skip
            if sum(care_matrix[idx]) == 0:
                # there's only one class present
                # which means there are no true positives/false positives
                #  and it's not usable for ROC-AUC
                # save this index to skip for later
                indices_to_skip.append(idx)

            idx += 1

    with open(care_matrix_filename, 'wb') as f1:
        pickle.dump(care_matrix, f1)

    with open(indices_filename, 'wb') as f1:
        pickle.dump(indices_to_skip, f1)

    return care_matrix, indices_to_skip


# return clean y_true and y_preds:
# entries with only one class in y_true are skipped (in both matrices), since they can't be used in AUC-ROC
def clean_care_matrices(care_true_matrix,
                        care_preds_matrix,
                        skip_indices_no_tp,
                        skip_indices_no_fp):
    skip_indices = list(set(skip_indices_no_tp + skip_indices_no_fp))
    new_care_true = np.delete(care_true_matrix, skip_indices, axis=0)
    new_care_preds = np.delete(care_preds_matrix, skip_indices, axis=0)

    return new_care_true, new_care_preds


# concat each row of y_true with y_preds to we can reliably shuffle it later
def create_data(y_true, y_preds):
    data = np.concatenate((y_preds, y_true), axis=1)
    nr_labels = int(data.shape[1]/2)

    return data, nr_labels


phen_codes = pickle.load(open('../resources/vocab_index_to_phen_task_nr.dict', 'rb'))

care_true, indices_to_skip_no_tp = get_care_matrix(
    '../pat_data/care_true.pkl',
    # this file will be created when you run evaluate_from_checkpoint.py on the test data
    # it was too big to include in the repo
    '../data/encodings/test/target.csv',
    phen_codes,
    '../pat_data/indices_to_skip_no_tp.pkl'
)

care_preds, indices_to_skip_no_fp = get_care_matrix(
    '../pat_data/care_preds.pkl',
    # this file will be created when you run evaluate_from_checkpoint.py on the test data
    # it was too big to include in the repo
    '../data/encodings/test/predictions.csv',
    phen_codes,
    '../pat_data/indices_to_skip_no_fp.pkl'
)

print('nr indices to skip in target', len(indices_to_skip_no_tp))
print('nr indices to skip in pred', len(indices_to_skip_no_fp))

print('orig care_true shape', care_true.shape)
print('orig care_preds shape', care_preds.shape)

care_true, care_preds = clean_care_matrices(
    care_true,
    care_preds,
    indices_to_skip_no_tp,
    indices_to_skip_no_fp
)

print('new care_true shape', care_true.shape)
print('new care_preds shape', care_preds.shape)


data, nr_labels = create_data(care_true, care_preds)

ret = print_metrics_multilabel(
    y_true=data[:, nr_labels:],
    predictions=data[:, :nr_labels],
    verbose=0,
)

# code from Benchmark study
# https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/evaluation/evaluate_pheno.py
metrics_list = [
    ('Macro ROC AUC', 'ave_auc_macro'),
    ('Micro ROC AUC', 'ave_auc_micro'),
    ('Weighted ROC AUC', 'ave_auc_weighted')
]

iters = 10000
n_tasks = 25
results = dict()
results['n_iters'] = iters

for (m, k) in metrics_list:
    results[m] = dict()
    results[m]['value'] = ret[k]
    results[m]['runs'] = []

for i in range(1, n_tasks + 1):
    m = 'ROC AUC of task {}'.format(i)
    results[m] = dict()
    results[m]['value'] = print_metrics_binary(data[:, n_tasks + i - 1], data[:, i - 1], verbose=0)['auroc']
    results[m]['runs'] = []

for iteration in range(iters):
    print(str(iteration), ' / ', str(iters))
    cur_data = sk_utils.resample(data, n_samples=len(data))
    ret = print_metrics_multilabel(care_true, care_preds, verbose=0)
    for (m, k) in metrics_list:
        results[m]['runs'].append(ret[k])
    for i in range(1, n_tasks + 1):
        m = 'ROC AUC of task {}'.format(i)
        cur_auc = print_metrics_binary(cur_data[:, n_tasks + i - 1], cur_data[:, i - 1], verbose=0)['auroc']
        results[m]['runs'].append(cur_auc)

reported_metrics = [m for m, k in metrics_list]
reported_metrics += ['ROC AUC of task {}'.format(i) for i in range(1, n_tasks + 1)]

for m in reported_metrics:
    runs = results[m]['runs']
    results[m]['mean'] = np.mean(runs)
    results[m]['median'] = np.median(runs)
    results[m]['std'] = np.std(runs)
    results[m]['2.5% percentile'] = np.percentile(runs, 2.5)
    results[m]['97.5% percentile'] = np.percentile(runs, 97.5)
    del results[m]['runs']

save_file = '../pat_data/pheno.json'
print("Saving the results (including task specific metrics) in {} ...".format(save_file))
with open(save_file, 'w') as f:
    json.dump(results, f)

print("Printing the summary of results (task specific metrics are skipped) ...")
for i in range(1, n_tasks + 1):
    m = 'ROC AUC of task {}'.format(i)
    del results[m]
    
print(results)
