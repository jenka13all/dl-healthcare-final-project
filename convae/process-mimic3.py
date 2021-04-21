import pandas as pd
import yaml
import csv
import argparse

parser = argparse.ArgumentParser(description='Create cohort vocabulary from MIMIC-III CSV diagnosis and item files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
args, _ = parser.parse_known_args()


def create_vocab(data_path):
    """
    creates a CSV file, data/cohort-vocab.csv, from mimic3 D_ICD_DIAGNOSES.csv and D_ITEMS.csv
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

    # populate items
    d_items = pd.read_csv(data_path+'D_ITEMS.csv', usecols=['LINKSTO', 'ITEMID', 'LABEL'], dtype=str)
    vocab_items = pd.DataFrame(
        data={
            'LABEL': d_items['LINKSTO'].astype(str) + '_' + d_items['ITEMID'].astype(str),
            'CODE': d_items['ITEMID'].astype(str),
            'DESC': '"' + d_items['LABEL'].str.replace('"', '') + '"'
        },
        dtype=pd.StringDtype()
    )

    vocab = pd.concat([vocab_diagnoses, vocab_items], ignore_index=True)

    # group the vitals into pre-specified 17 categories from Benchmark
    # read grouped vitals yaml
    with open('resources/vitals-map.yaml') as f:
        vitals_map = yaml.load(f, Loader=yaml.FullLoader)

    # update label for each grouped vital, add meaningful description
    for key, values_dict in vitals_map.items():
        item_ids = values_dict['item_ids']
        item_ids = [str(x) for x in item_ids]
        new_label = values_dict['vocab_label']

        # make a grouped label for the 17 chartevents
        vocab_to_update = vocab.CODE.isin(item_ids) & vocab.LABEL.str.match('chartevents_')
        vocab.loc[vocab_to_update, 'LABEL'] = new_label
        vocab.loc[vocab_to_update, 'DESC'] = '"' + key + '"'

    vocab.to_csv('data/cohort-vocab.csv',
                 quoting=csv.QUOTE_NONE,
                 escapechar='\\',
                 index_label='INDEX')


def map_vitals():
    """
    maps vitals in vitals-map.csv to grouped codes in itemid_to_variable_map.csv
    writes results as a yaml file data/vitals-map.yaml
    :return: None
    """
    vitals = pd.read_csv('resources/vitals-map.csv', quotechar='"')
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

    with open('resources/vitals-map.yaml', 'w') as file:
        documents = yaml.dump(vitals_items_dict, file)


database_path = args.mimic3_path
map_vitals()
create_vocab(database_path)
