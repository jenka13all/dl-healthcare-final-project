import pickle
import yaml
from collections import defaultdict
import os

'''
maps "task number" of care conditions to phenotype codes
'''


def mask_task_nr_to_codes(resources_path, filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))

    with open(resources_path + 'hcup_ccs_2015_definitions_benchmark.yaml', 'r') as f:
        definitions = yaml.load(f, Loader=yaml.FullLoader)

    with open(resources_path + 'task_nr_to_phenotype_label.dict', 'rb') as f1:
        task_to_phen_dict = pickle.load(f1)

    task_nr_to_phenotype_codes = defaultdict(list)
    for task_nr, phen_label in task_to_phen_dict.items():
        task_nr_to_phenotype_codes[task_nr] = definitions[phen_label]['codes']

    with open(filename, 'wb') as f1:
        pickle.dump(task_nr_to_phenotype_codes, f1)


resources_path = '../../common_resources/'
task_nr_to_phenotype_code = mask_task_nr_to_codes(
    resources_path,
    resources_path + 'task_nr_to_phenotype_codes.dict'
)

print(task_nr_to_phenotype_code)

