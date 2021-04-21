import pandas as pd
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Create patient EHR sequences (from diagnoses only).')
parser.add_argument('mimic3_pat_path', type=str, help='Root directory containing processed MIMIC-III patient files.')
parser.add_argument('cohort_type', type=str, help='Cohort type: test or train', default='train')
args, _ = parser.parse_known_args()

cohort_type = args.cohort_type
mimic3_pat_path = os.path.join(args.mimic3_pat_path, cohort_type)

if cohort_type == 'train':
    cohort_path = 'data/cohort-ehrseq.csv'
else:
    cohort_path = 'data/cohort_test-ehrseq.csv'

# make vocab a df
vocab_path = 'data/cohort-vocab.csv'
vocab = pd.read_csv(vocab_path)

# create diagnosis sequence file
seq_file = open(cohort_path, 'a+')

# if it's empty we need to add the headers
if os.stat(cohort_path).st_size == 0:
    seq_file.write('MRN,EHRseq' + "\n")

# keep track of subjects already written
# so we can pick up where we left off if necessary
tracking_file = 'data/cohort-subject-done.txt'
if not os.path.isfile(tracking_file):
    f = open(tracking_file, 'w')
    f.close()

with open(tracking_file, 'r') as f1:
    already_done = [line.rstrip('\n') for line in f1]

for subject_dir in tqdm(os.listdir(mimic3_pat_path), desc='Iterating over subjects'):
    # don't duplicate: there is only ONE diagnosis file per subject
    if subject_dir in already_done:
        print(subject_dir, 'already done')
        continue

    seq = 'pat_' + str(subject_dir) + ','
    dn = os.path.join(mimic3_pat_path, subject_dir)

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

    # log ones we've already done, so we can restart the loop again if necessary and not repeat work
    with open(tracking_file, 'a+') as done_file:
        done_file.write(subject_dir + "\n")

seq_file.close()

