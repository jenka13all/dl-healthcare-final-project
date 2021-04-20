import pandas as pd
import yaml


def create_vocab(data_path):
    """
    creates a CSV file, data/cohort-vocab.csv, from mimic3 D_ICD_DIAGNOSES.csv and D_ITEMS.csv
    COLUMNS: LABEL, CODE
    :param data_path: string
    :return: None
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


def map_vitals():
    """
    maps vitals in vitals-map.csv to grouped codes in itemid_to_variable_map.csv
    writes results as a yaml file data/vitals-map.yaml
    :return: None
    """
    vitals = pd.read_csv('data/vitals-map.csv', quotechar='"')
    item_id_to_variables = pd.read_csv('resources/itemid_to_variable_map.csv')

    vitals_items_dict = {}
    i = 1
    for label in vitals['Variable']:
        vitals_record = vitals[vitals['Variable'] == label]
        item_record = item_id_to_variables[item_id_to_variables['LEVEL2'] == label]

        itemids = list(item_record['ITEMID'])
        event_table = vitals_record['MIMIC-III table'].values[0]
        impute_value = vitals_record['Impute value'].values[0]
        modeled_as = vitals_record['Modeled as'].values[0]

        if i < 10:
            vitals_group = '0' + str(i)
        else:
            vitals_group = str(i)

        vitals_items_dict[label] = {
            'vocab_label': 'vitals_group_' + vitals_group,
            'event_table': event_table,
            'impute_value': impute_value,
            'modeled_as': modeled_as,
            'item_ids': itemids
        }

        i += 1

    with open('data/vitals-map.yaml', 'w') as file:
        documents = yaml.dump(vitals_items_dict, file)


def group_vitals():
    """
    groups chartevent codes for the 17 Benchmark vital signs within the vocabulary
    creates a new, grouped vocabulary file
    :return: None
    """
    # read cohort-vocab
    vocab = pd.read_csv('data/cohort-vocab.csv')

    # read vital yaml
    with open('data/vitals-map.yaml') as f:
        vitals_map = yaml.load(f, Loader=yaml.FullLoader)

    for key, values_dict in vitals_map.items():
        item_ids = values_dict['item_ids']
        item_ids = [str(x) for x in item_ids]
        new_label = values_dict['vocab_label']

        # we only want to update labels for chart events (vitals), NOT diagnoses
        vocab_to_update = vocab.CODE.isin(item_ids) & vocab.LABEL.str.match('chartevents_')
        vocab.loc[vocab_to_update, 'LABEL'] = new_label

    # write to an updated vocab file
    vocab.to_csv('data/cohort-vocab-grouped.csv', index=False)

# put your path to mimic3 data D_ITEMS.csv (chartevents) and D_ICD_DIAGNOSES.csv
#database_path = 'mimic-iii-clinical-database-1.4/'
#create_vocab(database_path)
#map_vitals()
#group_vitals()
