import pickle
import numpy as np


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return 'Key doesn\'t exist'


resource_path = '../resources/'

# Med2Vec patient medical codes (complete ICD9 codes)
pat_seqs = np.array(pickle.load(open('../Med2Vec_data/seqs.pkl', 'rb')), dtype=object)

# dictionaries mapping Med2Vec ICD9 codes to MIMIC-III ICD9 codes
med2vec_to_mimic3_seqs = pickle.load(open(resource_path + 'seqs_to_icd.dict', 'rb'))

# dictionary of the mapping of Med2Vec complete ICD9 codes (sequences) to Med2Vec IDs
with open(resource_path + 'processed.types', 'r') as f1:
    for line in f1:
        med2vec_code_to_id_seq = eval(line)

# dictionaries mapping "task number" to phenotype label and MIMIC-III ICD9 codes
task_nr_to_phen_label = pickle.load(open(resource_path + 'task_nr_to_phenotype_label.dict', 'rb'))
task_nr_to_phen_codes = pickle.load(open(resource_path + 'task_nr_to_phenotype_codes.dict', 'rb'))


def get_care_condition_for_med2vec_id(code):
    # get Med2Vec ICD9 code for Med2Vec ID
    med2vec_key = get_key(med2vec_code_to_id_seq, code)  # D_728.86

    # get MIMIC-III ICD9 code for Med2Vec ICD9 code
    mimic3_code = med2vec_to_mimic3_seqs[med2vec_key]  # 72886

    # does this code belong to one of the 25 care conditions from the Benchmark task
    for task_nr, phen_codes in task_nr_to_phen_codes.items():
        if mimic3_code in phen_codes:
            return med2vec_key, task_nr_to_phen_label[task_nr]


# sequences (complete ICD9 codes) for each patients' codes
for idx, seq in enumerate(pat_seqs):
    if seq == [-1]:
        # we're interested in the seq previous to this, since it's the last visit of a patient
        last_visit = pat_seqs[idx-1]

        for code in last_visit:
            print(get_care_condition_for_med2vec_id(code))

