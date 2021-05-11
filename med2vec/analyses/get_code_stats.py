import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


pat_labels = np.array(pickle.load(open('../Med2Vec_data/labels.pkl', 'rb')), dtype=object)

# get total codes over patients distribution:
# key = number of codes
# value = number of patients with that number of codes - across all visits
pat_codes = defaultdict(list)
pat_idx = 0
for visit in pat_labels:
    if visit == [-1]:
        pat_idx += 1
    else:
        for code in visit:
            # only add unique codes
            if code not in pat_codes[pat_idx]:
                pat_codes[pat_idx].append(code)

pat_codes_dist_dict = defaultdict(int)
for pat, codes in pat_codes.items():
    pat_nr_codes = len(codes)
    pat_codes_dist_dict[pat_nr_codes] += 1

# get total codes over visits distribution:
# key = number of codes
# value = number of VISITS with that number of codes
visit_codes = defaultdict(list)
visit_idx = 0
for visit in pat_labels:
    if visit != [-1]:
        visit_codes[visit_idx].append(visit)
    else:
        visit_idx += 1

visit_codes_dist_dict = defaultdict(int)
for visit, codes in visit_codes.items():
    visit_nr_codes = len(codes)
    visit_codes_dist_dict[visit_nr_codes] += 1

# histogram of pat-codes distribution
plt.bar(list(pat_codes_dist_dict.keys()), pat_codes_dist_dict.values(), color='g')
plt.savefig('../figures/pat_codes_distribution.png', dpi=700)

# histogram of visit-codes distribution
plt.bar(list(visit_codes_dist_dict.keys()), visit_codes_dist_dict.values(), color='g')
plt.savefig('../figures/visit_codes_distribution.png', dpi=700)

# print some basic statistics about the variance of the number of codes per patient
nr_codes_for_pat = pat_codes_dist_dict.keys()
print('maximum number of UNIQUE codes for a patient', max(nr_codes_for_pat))  # 90
print('minimum number of UNIQUE codes for a patient', min(nr_codes_for_pat))  # 1

# print some basic statistics about the variance of the number of codes per visit
nr_codes_for_visit = visit_codes_dist_dict.keys()
print('maximum number of codes for a visit', max(nr_codes_for_visit))  # 42
print('minimum number of codes for a visit', min(nr_codes_for_visit))  # 2

print('non-weighted average number of codes per patient', round(len(pat_labels)/len(pat_codes), 2))  # 3.65
print('non-weighted average number of codes per visit', round(len(pat_labels)/len(visit_codes), 2))  # 3.65
