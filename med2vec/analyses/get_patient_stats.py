import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


pat_seqs = np.array(pickle.load(open('../output/processed.seqs', 'rb')), dtype=object)
pat_labels = np.array(pickle.load(open('../output/processed.3digitICD9.seqs', 'rb')), dtype=object)

print('total visits', len(pat_seqs))

# count number of patients
pat_idx = 0
pat_visit_dict = defaultdict(list)
for seq in pat_seqs:
    if seq != [-1]:
        pat_visit_dict[pat_idx].append(seq)
    else:
        pat_idx += 1

# patient 1 visits
# print(pat_visit_dict[0])

# total patients
nr_patients = len(pat_visit_dict)
print('total patients:', nr_patients)

# get total visit distribution:
# key = number of visits
# value = number of patients with that number of visits
visit_dist_dict = defaultdict(int)
for patient, visits in pat_visit_dict.items():
    pat_nr_visits = len(visits)
    visit_dist_dict[pat_nr_visits] += 1

# histogram of visit number distribution
plt.bar(list(visit_dist_dict.keys()), visit_dist_dict.values(), color='g')
plt.savefig('../figures/visit_distribution.png', dpi=700)

# print some basic statistics about the variance of the number of visits
nr_visits = visit_dist_dict.keys()
print('maximum number of visits for a patient', max(nr_visits))
print('minimum number of visits for a patient', min(nr_visits))

print('non-weighted average number of visits per patient', round(len(pat_seqs)/len(pat_visit_dict), 2))

weighted_sum = []
nr_patients_list = []
for nr_visits, nr_patients in visit_dist_dict.items():
    weighted_sum.append(nr_visits * nr_patients)
    nr_patients_list.append(nr_patients)


print('weighted average number of visits per patient', round(sum(weighted_sum) / sum(nr_patients_list), 2))
