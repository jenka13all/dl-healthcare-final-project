import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt


def get_pearson_cc(file_path, study_name):
    results = pd.read_csv(file_path)
    x = np.array(results['Train'])
    y = np.array(results['AUC-ROC'])
    r, p = scipy.stats.pearsonr(x, y)

    print(study_name, 'correlation coefficient and p-value:', str(r), str(p))

    return x, y


bench_results = '../data/phenotyping/evaluation/results_table.csv'
bx, by = get_pearson_cc(bench_results, 'Benchmark')

plt.style.use('ggplot')

slope, intercept, r, p, stderr = scipy.stats.linregress(mx, my)
line = f'Pearson Correlation Coefficient: {r:.2f}, p-value: {p: 5f}'

fig, ax = plt.subplots()
ax.plot(bx, by, linewidth=0, marker='s', label='Care-conditions')
ax.plot(bx, intercept + slope * bx, label=line)
ax.set_xlabel('Care-condition Prevalence')
ax.set_ylabel('AUC-ROC score')
ax.legend(facecolor='white')

plt.savefig('../figures/prevalence_pearson.png', dpi=700)
