import pandas as pd
import os
from tqdm import tqdm


# make vocab a df
vocab_path = 'data/cohort-vocab.csv'
vocab = pd.read_csv(vocab_path)

mimic3_train = '../../mimic3-benchmarks/data/root/train'

subject_done = []
seq_file = open('data/cohort-ehrseq.csv', 'a+')
for subject_dir in tqdm(os.listdir(mimic3_train), desc='Iterating over subjects'):
    # don't duplicate: there is only ONE diagnosis file per subject
    if subject_dir not in subject_done:
        seq = 'pat_' + str(subject_dir) + ','
        dn = os.path.join(mimic3_train, subject_dir)

        # subject diagnoses
        diagnoses_file = os.path.join(dn, 'diagnoses.csv')
        diagnoses = pd.read_csv(diagnoses_file, usecols=['SUBJECT_ID', 'ICD9_CODE'], dtype=str)

        # get the index for the icd9 codes, add it to this subject's sequence
        icd9s = list(diagnoses['ICD9_CODE'])
        for icd9 in icd9s:
            seq_diagnoses = vocab.loc[vocab['LABEL'] == 'DIAGNOSIS_' + str(icd9)]

            # if can't find diagnosis, we want to know why
            if seq_diagnoses.shape[0] == 0:
                print(subject_dir, str(icd9))

            seq += str(seq_diagnoses['INDEX'].values[0]) + ','

        # remove last comma and append to file
        seq = seq[:-1]
        seq_file.write(seq + "\n")
        subject_done.append(str(subject_dir))

seq_file.close()
