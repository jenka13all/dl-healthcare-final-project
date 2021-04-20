import pandas as pd
import os
from tqdm import tqdm


vocab_path = '../data/cohort-vocab.csv'
vocab = pd.read_csv(vocab_path)

mimic3_train = '../../../mimic3-benchmarks/data/root/test'

seq_file = open('../data/cohort-ehrseq.csv', 'a+')
for subject_dir in tqdm(os.listdir(mimic3_train), desc='Iterating over subjects'):
    if subject_dir == '225':
        seq = 'pat_' + str(subject_dir) + ','
        dn = os.path.join(mimic3_train, subject_dir)

        # diagnoses
        diagnoses_file = os.path.join(dn, 'diagnoses.csv')
        diagnoses = pd.read_csv(diagnoses_file, usecols=['SUBJECT_ID', 'ICD9_CODE'])

        icd9s = list(diagnoses['ICD9_CODE'])
        for icd9 in icd9s:
            seq_diagnoses = vocab.loc[vocab['LABEL'] == 'DIAGNOSIS_' + str(icd9)]
            seq += str(seq_diagnoses['INDEX'].values[0]) + ','

        # remove last comma and append to file
        seq = seq[:-1]
        seq_file.write(seq + "\n")

seq_file.close()
should_be = 'pat_225,952,2137,5421,4416,5147'
