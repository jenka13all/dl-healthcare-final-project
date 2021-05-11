import pickle
import pandas as pd
from collections import defaultdict


phen_codes = pickle.load(open('../../common_resources/phenotype_codes_to_task_nr.dict', 'rb'))
cohort_vocab = pd.read_csv('../data/cohort_vocab_icd.csv', usecols=['INDEX', 'CODE'])

vocab_index_to_phen_task_nr = defaultdict(int)

for code, task_nr in phen_codes.items():
    vocab = cohort_vocab[cohort_vocab['CODE'] == code]
    index = vocab['INDEX'].values
    if len(index) == 0:
        continue
    vocab_index_to_phen_task_nr[index[0]] = task_nr

with open('../resources/vocab_index_to_phen_task_nr.dict', 'wb') as f1:
    pickle.dump(vocab_index_to_phen_task_nr, f1)
