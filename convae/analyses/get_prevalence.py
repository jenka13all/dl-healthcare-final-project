import pickle
import csv
from collections import defaultdict
import utils as ut


def get_condition_prevalence(pat_data_file, phen_codes_dict):
    total_patients = 0
    condition_counts_dict = defaultdict(int)
    with open('../data/'+pat_data_file) as f:
        rd = csv.reader(f)
        # skip header
        next(rd)
        for r in rd:
            # populate care_matrix
            seqs = r[1:]
            for seq in seqs:
                seq = int(seq)
                if seq in phen_codes_dict:
                    task_nr = phen_codes_dict[seq]
                    condition_counts_dict[task_nr] += 1

            total_patients += 1

    prevalence_dict = defaultdict(float)
    for task_nr, prev_total in condition_counts_dict.items():
        prevalence_dict[task_nr] = round(prev_total/total_patients, 3)

    return prevalence_dict


vocab_to_phen_dict = pickle.load(open('../resources/vocab_index_to_phen_task_nr.dict', 'rb'))

train = ut.dt_files['ehr-file']
test = ut.dt_files['ehr-file-test']

train_prevalence = get_condition_prevalence(train, vocab_to_phen_dict)
test_prevalence = get_condition_prevalence(test, vocab_to_phen_dict)

with open('../resources/train_prevalence.dict', 'wb') as f:
    pickle.dump(train_prevalence, f)

with open('../resources/test_prevalence.dict', 'wb') as f:
    pickle.dump(test_prevalence, f)
