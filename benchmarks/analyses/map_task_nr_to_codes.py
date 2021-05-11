import pickle
import yaml
from collections import defaultdict
import os

'''
maps "task number" of care conditions to phenotype codes
'''


def get_phenotype_labels(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))

    # map "task number" (e.g. "1", "2") to correct conditions
    labels_file = '../data/root/phenotype_labels.csv'
    labels = pd.read_csv(labels_file)
    labels = list(labels.head(1))

    phenotype_labels = {}
    for i in range(1, len(labels)+1):
        phenotype_labels[i] = labels[i-1]

    with open(filename, 'wb') as f1:
        pickle.dump(phenotype_labels, f1)

    return phenotype_labels


def mask_task_nr_to_codes(resources_path, filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))

    with open(resources_path + 'hcup_ccs_2015_definitions_benchmark.yaml', 'r') as f:
        definitions = yaml.load(f, Loader=yaml.FullLoader)

    task_to_phen_dict = get_phenotype_labels(resources_path + 'task_nr_to_phenotype_label.dict')

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
