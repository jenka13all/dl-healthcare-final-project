import pandas as pd
import yaml
import json
import pickle
import os

'''
maps the "task number" in the results JSON to care condition name
formats the results JSON into tabular format and saves as csv file
'''


def convert_json_results_to_dict(results_json, phenotype_labels):
    with open(results_json) as json_file:
        results = json.load(json_file)

    results_dict = {}
    for i in range(1, 26):
        key = "ROC AUC of task {}".format(i)
        results_dict[i] = results[key]

    # get the NAME of "Task x" (the care condition) along with its results
    final_results_dict = {}
    for i in range(1, 26):
        key = phenotype_labels[i]
        final_results_dict[key] = results_dict[i]

    return final_results_dict


def get_results_table(filename, final_results_dict, train_dict, test_dict, definitions):
    if os.path.isfile(filename):
        return pd.read_csv(filename)

    # build final dataframe of care condition, type, prevalence over all visits in split, AUC-ROC
    metrics_df = pd.DataFrame(columns=['Phenotype', 'Type', 'Train', 'Test', 'AUC-ROC'])
    i = 1
    for rx, metrics in final_results_dict.items():
        prev_train = train_dict[i]
        prev_test = test_dict[i]

        auc_roc = round(metrics['value'], 3)
        rx_type = definitions[rx]['type']

        metrics_df.loc[i] = [rx, rx_type, prev_train, prev_test, auc_roc]
        i += 1

    # save final dataframe to CSV
    metrics_df.to_csv(filename, float_format='%.3f')

    return metrics_df


local_resources_path = '../resources/'
common_resources_path = '../../common_resources/'

with open(common_resources_path + 'hcup_ccs_2015_definitions_benchmark.yaml', 'r') as f:
    definitions = yaml.load(f, Loader=yaml.FullLoader)

phenotype_labels = pickle.load(open(common_resources_path + 'task_nr_to_phenotype_label.dict', 'rb'))

# open pheno_results.json
results_json = '../pat_data/pheno.json'
final_results_dict = convert_json_results_to_dict(results_json, phenotype_labels)

# get prevalence of each care-condition in train and test population
test = pickle.load(open(local_resources_path + 'train_prevalence.dict', 'rb'))
train = pickle.load(open(local_resources_path + 'test_prevalence.dict', 'rb'))

results = get_results_table(
    '../pat_data/results_table.csv',
    final_results_dict,
    train,
    test,
    definitions
)

print(results)
