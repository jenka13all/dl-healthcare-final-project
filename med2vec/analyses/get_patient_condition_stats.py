from collections import defaultdict
import numpy as np
import pickle

'''
gets prevalence of each care condition in training and test sets
and writes this to dictionaries in "resources" for later use
'''


def get_med2vec_to_id_dict(resource_path):
    # dictionary of the mapping of Med2Vec complete ICD9 codes (sequences) to Med2Vec IDs
    with open(resource_path + 'processed.types', 'r') as f1:
        for line in f1:
            med2vec_code_to_id_seq = eval(line)

    return med2vec_code_to_id_seq


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return 'Key doesn\'t exist'


def get_care_condition_for_med2vec_id(code):
    # get Med2Vec ICD9 code for Med2Vec ID
    med2vec_key = get_key(med2vec_code_to_id_seq, code)  # D_728.86

    # get MIMIC-III ICD9 code for Med2Vec ICD9 code
    mimic3_code = med2vec_to_mimic3_seqs[med2vec_key]  # 72886

    # does this code belong to one of the 25 care conditions from the Benchmark task
    for task_nr, phen_codes in task_nr_to_phen_codes.items():
        if mimic3_code in phen_codes:
            return task_nr

    return None


# patient medical codes: training and test
data_path = '../tensorflow-implementation/Med2Vec_data/'
pat_seqs = np.array(pickle.load(open(data_path + 'seqs.pkl', 'rb')), dtype=object)
train_pat_seqs = np.array(pickle.load(open(data_path + 'train_seqs.pkl', 'rb')), dtype=object)
test_pat_seqs = np.array(pickle.load(open(data_path + 'test_seqs.pkl', 'rb')), dtype=object)
two_pat_seqs = np.array(pickle.load(open(data_path + 'test_2pat_data/seqs.pkl', 'rb')), dtype=object)

resource_path = '../resources/'

# dictionaries mapping "task number" to phenotype label and MIMIC-III ICD9 codes
task_nr_to_phen_label = pickle.load(open(resource_path + 'task_nr_to_phenotype_label.dict', 'rb'))
task_nr_to_phen_codes = pickle.load(open(resource_path + 'task_nr_to_phenotype_codes.dict', 'rb'))

# dictionary of Med2Vec ICD code to Med2Vec ID
med2vec_code_to_id_seq = get_med2vec_to_id_dict(resource_path)

# dictionary mapping Med2Vec IDs to MIMIC-3 ICD codes
med2vec_to_mimic3_seqs = pickle.load(open(resource_path + 'seqs_to_icd.dict', 'rb'))


def get_care_condition_prevalence(pat_seqs):
    prevalence_count_dict = defaultdict(int)
    pat_idx = 0
    pat_conditions = {pat_idx: []}
    for seq in pat_seqs:
        if seq == [-1]:
            # new patient
            pat_idx += 1
            pat_conditions[pat_idx] = []
        else:
            for code in seq:
                task_nr = get_care_condition_for_med2vec_id(code)
                # only add to total prevalence if unique for this patient (i.e. no duplicates per patient)
                if task_nr is not None and task_nr not in pat_conditions[pat_idx]:
                    pat_conditions[pat_idx].append(task_nr)
                    prevalence_count_dict[task_nr] += 1

    total_patients = pat_idx + 1

    prevalence_dict = {}
    for task, total in prevalence_count_dict.items():
        prevalence_dict[task] = round(total / total_patients, 3)

    return prevalence_dict


train_prevalence_dict = get_care_condition_prevalence(train_pat_seqs)
test_prevalence_dict = get_care_condition_prevalence(test_pat_seqs)

with open(resource_path + 'care_condition_train_prevalence.dict', 'wb') as f1:
    pickle.dump(train_prevalence_dict, f1)

with open(resource_path + 'care_condition_test_prevalence.dict', 'wb') as f1:
    pickle.dump(test_prevalence_dict, f1)
