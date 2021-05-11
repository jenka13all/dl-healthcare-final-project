import pandas as pd
import argparse
import csv

parser = argparse.ArgumentParser(description='Create cohort vocabulary from MIMIC-III CSV diagnosis and item files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
args, _ = parser.parse_known_args()


def create_vocab(data_path):
    """
    creates a CSV file, data/cohort_vocab.csv, from mimic3:
    D_ICD_DIAGNOSES.csv (ICD9 diagnoses)
    COLUMNS: LABEL, CODE, DESC
    :param data_path: string
    :return: None
    """

    # populate diagnoses
    d_diagnoses = pd.read_csv(data_path + '/D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE', 'LONG_TITLE'], dtype=str)
    vocab_diagnoses = pd.DataFrame(
        data={
            'LABEL': 'DIAGNOSIS_' + d_diagnoses['ICD9_CODE'].astype(str),
            'CODE': d_diagnoses['ICD9_CODE'].astype(str),
            'DESC': '"' + d_diagnoses['LONG_TITLE'].str.replace('"', '') + '"'
        },
        dtype=pd.StringDtype()
    )

    # write cohort-vocab to CSV
    vocab_diagnoses.to_csv(
        'data/cohort_vocab_icd.csv',
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
        index_label='INDEX'
    )

    return len(vocab_diagnoses)


database_path = args.mimic3_path
total_vocab = create_vocab(database_path)

print('total vocabulary of ', str(total_vocab), ' medical concepts')

# python3 create_mimic3_vocab_only_icd.py mimic-iii-clinical-database-1.4