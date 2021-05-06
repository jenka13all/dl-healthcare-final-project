import numpy as np
import scipy.stats
import pandas as pd


def get_pearson_cc(file_path, study_name):
    results = pd.read_csv(file_path)
    x = np.array(results['Train'])
    y = np.array(results['AUC-ROC'])
    r, p = scipy.stats.pearsonr(x, y)

    print(study_name, 'correlation coefficient and p-value:', str(r), str(p))


bench_results = '../../benchmarks/data/phenotyping/evaluation/results_table.csv'
get_pearson_cc(bench_results, 'Benchmark')

med2vec_results = '../resources/results_table.csv'
get_pearson_cc(med2vec_results, 'Med2Vec')
