import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Create cohort vocabulary from MIMIC-III CSV diagnosis and item files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
args, _ = parser.parse_known_args()


def create_vocab(data_path):
    """
    creates a CSV file, data/cohort-vocab.csv, from mimic3:
    D_ICD_DIAGNOSES.csv (ICD9 diagnoses)
    D_ITEMS.csv (microbiology events only)
    D_LABITEMS.csv (lab events)
    TODO: update with
      PRESCRIPTIONS.csv (medications, need to be normalized to RxNorm)
      D_CPT.csv (CPT-4 procedure codes)
    COLUMNS: LABEL, CODE, DESC
    :param data_path: string
    :return: None
    """

    # populate diagnoses
    d_diagnoses = pd.read_csv(data_path+'D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE', 'LONG_TITLE'], dtype=str)
    vocab_diagnoses = pd.DataFrame(
        data={
            'LABEL': 'DIAGNOSIS_' + d_diagnoses['ICD9_CODE'].astype(str),
            'CODE': d_diagnoses['ICD9_CODE'].astype(str),
            'DESC': '"' + d_diagnoses['LONG_TITLE'].str.replace('"', '') + '"'
        },
        dtype=pd.StringDtype()
    )

    # populate microbiology
    d_items = pd.read_csv(data_path+'D_ITEMS.csv', usecols=['LINKSTO', 'ITEMID', 'LABEL'], dtype=str)
    d_items = d_items[d_items['LINKSTO'] == 'microbiologyevents']
    vocab_items = pd.DataFrame(
        data={
            'LABEL': 'MICROB_' + d_items['ITEMID'].astype(str),
            'CODE': d_items['ITEMID'].astype(str),
            'DESC': '"' + d_items['LABEL'].str.replace('"', '') + '"'
        },
        dtype=pd.StringDtype()
    )

    # populate lab
    d_lab = pd.read_csv(
        data_path+'D_LABITEMS.csv',
        usecols=['LOINC_CODE', 'ITEMID', 'LABEL', 'FLUID', 'CATEGORY'],
        dtype=str
    )
    d_lab = d_lab[d_lab['LOINC_CODE'] != '']
    vocab_lab = pd.DataFrame(
        data={
            'LABEL': d_lab['LOINC_CODE'].astype(str),
            'CODE': d_lab['ITEMID'].astype(str),
            'DESC': '"' + d_lab['LABEL'].str.replace('"', '') +
                    '-' + d_lab['FLUID'].str.replace('"', '') +
                    '-' + d_lab['CATEGORY'].str.replace('"', '') + '"'
        },
        dtype=pd.StringDtype()
    )

    # concatenate all vocab parts
    vocab = pd.concat([vocab_diagnoses, vocab_items, vocab_lab], ignore_index=True)

    # write cohort-vocab to CSV
    vocab.to_csv(
        'data/cohort-vocab.csv',
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
        index_label='INDEX'
    )

    return len(vocab)


database_path = args.mimic3_path
total_vocab = create_vocab(database_path)

print('total vocabulary of ', str(total_vocab), ' medical concepts')
