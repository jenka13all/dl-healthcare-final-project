import pickle
import numpy as np


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return 'Key doesn\'t exist'


resource_path = '../resources/'

# test set
seqs_pat1_visit1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels_pat1_visit1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Med2Vec patient medical codes and labels
pat_seqs = np.array(pickle.load(open('../Med2Vec_data/seqs.pkl', 'rb')), dtype=object)
pat_labels = np.array(pickle.load(open('../Med2Vec_data/labels.pkl', 'rb')), dtype=object)

# dictionaries mapping Med2Vec ICD9 codes to MIMIC-III ICD9 codes
med2vec_to_mimic3_seqs = pickle.load(open(resource_path + 'seqs_to_icd.dict', 'rb'))
med2vec_to_mimic3_labels = pickle.load(open(resource_path + 'labels_to_icd.dict', 'rb'))

# dictionary of the mapping of Med2Vec complete ICD9 codes (sequences) to Med2Vec IDs
with open(resource_path + 'processed.types', 'r') as f1:
    for line in f1:
        med2vec_code_to_id_seq = eval(line)

# dictionary of the mapping of Med2Vec 3-digit ICD9 codes (labels) to Med2Vec IDs
with open(resource_path + 'processed.3digitICD9.types', 'r') as f1:
    for line in f1:
        med2vec_code_to_id_label = eval(line)

# dictionaries mapping "task number" to phenotype label and MIMIC-III ICD9 codes
task_nr_to_phen_label = pickle.load(open(resource_path + 'task_nr_to_phenotype_label.dict', 'rb'))
task_nr_to_phen_codes = pickle.load(open(resource_path + 'task_nr_to_phenotype_codes.dict', 'rb'))

# good news: all codes (ints mapping to ICD9s) are unique in processed.seqs and in processed.3digitICD9.seqs
# bad news: 3-digit ICD9 labels do not map to unique care conditions
# have to retrain model using seqs.pkl as seqs AND labels

# sequences (complete ICD9 codes) for each patients' codes
for seq in seqs_pat1_visit1:
    # get Med2Vec ICD9 code for Med2Vec ID
    med2vec_key = get_key(med2vec_code_to_id_seq, seq)  # D_728.86

    # get MIMIC-III ICD9 code for Med2Vec ICD9 code
    mimic3_code = med2vec_to_mimic3_seqs[med2vec_key]

    # does this code belong to one of the 25 care conditions from the Benchmark task
    for task_nr, phen_codes in task_nr_to_phen_codes.items():
        if mimic3_code in phen_codes:
            print(seq, med2vec_key, mimic3_code, task_nr)

# labels (3-digit ICD-9 codes) for each patients' codes
for label in labels_pat1_visit1:
    # get Med2Vec ICD9 code for Med2Vec ID
    med2vec_key = get_key(med2vec_code_to_id_label, label)  # D_728.86

    # get MIMIC-III ICD9 code for Med2Vec ICD9 code
    # for the labels, each 3-digit ICD code maps to multiple MIMIC-3 codes
    mimic3_codes = med2vec_to_mimic3_labels[med2vec_key]

    # does this code belong to one of the 25 care conditions from the Benchmark task
    for task_nr, phen_codes in task_nr_to_phen_codes.items():
        for mimic3_code in mimic3_codes:
            if mimic3_code in phen_codes:
                print(label, med2vec_key, mimic3_code, task_nr)
