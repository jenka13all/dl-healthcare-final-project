import numpy as np
from sklearn import metrics
import pickle
import sklearn.utils as sk_utils
import json


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
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


# concat each row of y_true with y_preds to we can reliably shuffle it later
def create_data(y_true, y_preds):
    data = np.concatenate((y_preds, y_true), axis=1)
    nr_labels = int(data.shape[1]/2)

    return data, nr_labels


model_path = '../Med2Vec_model/train_test_split_3_digit_icd/'

# get care_true and care_preds matrices created during visit prediction of Med2VecRunner.py
care_true = pickle.load(open(model_path + 'care_true.pkl', 'rb'))
care_preds = pickle.load(open(model_path + 'care_preds.pkl', 'rb'))
data, nr_labels = create_data(care_true, care_preds)

ret = print_metrics_multilabel(y_true=data[:, nr_labels:], predictions=data[:, :nr_labels], verbose=0)

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

save_file = model_path + 'pheno.json'
print("Saving the results (including task specific metrics) in {} ...".format(save_file))
with open(save_file, 'w') as f:
    json.dump(results, f)

print("Printing the summary of results (task specific metrics are skipped) ...")
for i in range(1, n_tasks + 1):
    m = 'ROC AUC of task {}'.format(i)
    del results[m]
print(results)
