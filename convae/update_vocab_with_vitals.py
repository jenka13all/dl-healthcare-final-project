import pandas as pd
import yaml

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
