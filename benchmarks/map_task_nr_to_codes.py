import pickle
import yaml
from collections import defaultdict

resources_path = 'resources/'

with open(resources_path + 'hcup_ccs_2015_definitions_benchmark.yaml', 'r') as f:
    definitions = yaml.load(f, Loader=yaml.FullLoader)

with open(resources_path + 'task_nr_to_phenotype_label.dict', 'rb') as f1:
    task_to_phen_dict = pickle.load(f1)

task_nr_to_phenotype_codes = defaultdict(list)
for task_nr, phen_label in task_to_phen_dict.items():
    task_nr_to_phenotype_codes[task_nr] = definitions[phen_label]['codes']

with open(resources_path + 'task_nr_to_phenotype_codes.dict', 'wb') as f1:
    pickle.dump(task_nr_to_phenotype_codes, f1)
