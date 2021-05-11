import pandas as pd
import os
from tqdm import tqdm
import argparse
import utils as ut

parser = argparse.ArgumentParser(description='Create patient EHR sequences.')
parser.add_argument('mimic3_pat_path', type=str, help='Root directory containing processed MIMIC-III patient files.')
parser.add_argument('cohort_type', type=str, help='Cohort type: test or train', default='train')
args, _ = parser.parse_known_args()

cohort_type = args.cohort_type
mimic3_pat_path = os.path.join(args.mimic3_pat_path, cohort_type)

if cohort_type == 'train':
    cohort_path = os.path.join('data', ut.dt_files['ehr-file'])
    tracking_file = os.path.join('data', 'icd_cohort_subject_done.txt')
else:
    cohort_path = os.path.join('data', ut.dt_files['ehr-file-test'])
    tracking_file = os.path.join('data', 'icd_test_cohort_subject-done.txt')

# make vocab a df
vocab_path = os.path.join('data', ut.dt_files['vocab'])
vocab = pd.read_csv(vocab_path)

# create diagnosis sequence file
seq_file = open(cohort_path, 'a+')

# if it's empty we need to add the headers
if os.stat(cohort_path).st_size == 0:
    seq_file.write('MRN,EHRseq' + "\n")

# keep track of subjects already written
# so we can pick up where we left off if necessary
if not os.path.isfile(tracking_file):
    f = open(tracking_file, 'w')
    f.close()

with open(tracking_file, 'r') as f1:
    already_done = [line.rstrip('\n') for line in f1]

for subject_dir in tqdm(os.listdir(mimic3_pat_path), desc='Iterating over subjects'):
    # don't duplicate: there is only ONE diagnosis and ONE event file per subject
    if subject_dir in already_done:
        print(subject_dir, 'already done')
        continue

    seq = 'pat_' + str(subject_dir) + ','
    dn = os.path.join(mimic3_pat_path, subject_dir)

    # get order of stays in case where there's more than one
    stays_file = os.path.join(dn, 'stays.csv')
    stays = pd.read_csv(
        stays_file,
        usecols=['HADM_ID', 'INTIME'],
        parse_dates=['INTIME'],
        infer_datetime_format=True
    )

    sorted_admissions = stays.sort_values(by='INTIME')['HADM_ID'].tolist()

    # subject diagnoses
    diagnoses_file = os.path.join(dn, 'diagnoses.csv')
    diagnoses = pd.read_csv(
        diagnoses_file,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
        dtype=str
    )

    # we want diagnoses in order of admission date
    for adm_id in sorted_admissions:
        adm_diagnoses = diagnoses[diagnoses['HADM_ID'] == str(adm_id)]

        # get the index for the icd9 codes, add it to this subject's sequence
        icd9s = list(adm_diagnoses['ICD9_CODE'])
        for icd9 in icd9s:
            seq_diagnoses = vocab.loc[vocab['LABEL'] == 'DIAGNOSIS_' + str(icd9)]

            # if can't find diagnosis, we want to know why
            if seq_diagnoses.shape[0] == 0:
                print('could not find diagnosis for subj:', subject_dir, str(icd9))
                continue

            seq += str(seq_diagnoses['INDEX'].values[0]) + ','

    # remove last comma and append to file
    seq = seq[:-1]
    seq_file.write(seq + "\n")

    # log ones we've already done, so we can restart the loop again if necessary and not repeat work
    with open(tracking_file, 'a+') as done_file:
        done_file.write(subject_dir + "\n")

seq_file.close()

# python3 create_patient_seqs_only_icd.py pat_data/root train
# python3 create_patient_seqs_only_icd.py pat_data/root test
