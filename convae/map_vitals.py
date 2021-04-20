import pandas as pd
import yaml

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