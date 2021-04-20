import pandas as pd

# put your path to mimic3 data D_ITEMS.csv (chartevents) and D_ICD_DIAGNOSES.csv
data_path = 'mimic-iii-clinical-database-1.4/'


def create_vocab(data_path):
    """
    creates a CSV file, data/cohort-vocab.csv, from mimic3 D_ICD_DIAGNOSES.csv and D_ITEMS.csv
    COLUMNS:
    LABEL, CODE
    """
    d_diagnoses = pd.read_csv(data_path+'D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE'], dtype=str)
    vocab_diagnoses = pd.DataFrame(
        data={
            'LABEL': 'DIAGNOSIS_' + d_diagnoses['ICD9_CODE'].astype(str),
            'CODE': d_diagnoses['ICD9_CODE'].astype(str)
        },
        dtype=pd.StringDtype()
    )

    d_items = pd.read_csv(data_path+'D_ITEMS.csv', usecols=['LINKSTO', 'ITEMID'], dtype=str)
    vocab_items = pd.DataFrame(
        data={
            'LABEL': d_items['LINKSTO'].astype(str) + '_' + d_items['ITEMID'].astype(str),
            'CODE': d_items['ITEMID'].astype(str)
        },
        dtype=pd.StringDtype()
    )

    vocab = pd.concat([vocab_diagnoses, vocab_items])
    vocab.to_csv('data/cohort-vocab.csv', index=False)


# vocab = pd.read_csv('data/cohort-vocab.csv')
# print(vocab.shape)
