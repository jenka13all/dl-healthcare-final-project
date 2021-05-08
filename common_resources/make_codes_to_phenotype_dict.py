import pickle
from collections import defaultdict

# we want a dictionary with ICD9 code as key, corresponding phenotype task number as value
phen_codes_dict = pickle.load(open('task_nr_to_phenotype_codes.dict', 'rb'))

phenotype_codes_to_task_nr = defaultdict(int)
for task_nr, codes in phen_codes_dict.items():
    for code in codes:
        phenotype_codes_to_task_nr[code] = task_nr


with open('phenotype_codes_to_task_nr.dict', 'wb') as f1:
    pickle.dump(phenotype_codes_to_task_nr, f1)
