import pandas as pd
import yaml
import json

'''
map the task number to diagnosis name
format phenotyping AUC-ROC results into tabular format and save
'''

data_path = '../data/phenotyping/'
resources_path = '../resources/'

with open(resources_path + 'hcup_ccs_2015_definitions_benchmark.yaml', 'r') as f:
    definitions = yaml.load(f, Loader=yaml.FullLoader)

# map "task number" (e.g. "1", "2") to correct conditions
labels_file = '../data/root/phenotype_labels.csv'
labels = pd.read_csv(labels_file)
labels = list(labels.head(1))

phenotype_labels = {}
for i in range(1, len(labels)+1):
    phenotype_labels[i] = labels[i-1]

# open pheno_results.json
results_json = data_path + 'evaluation/pheno_results.json'

with open(results_json) as json_file:
    results = json.load(json_file)

results_dict = {}
for i in range(1, 26):
    key = "ROC AUC of task {}".format(i)
    results_dict[i] = results[key]

#print(results_dict)
# get the NAME of "Task 1" (the diagnosis) along with its results
final_results_dict = {}
for i in range(1, 26):
    key = phenotype_labels[i]
    final_results_dict[key] = results_dict[i]

# build final dataframe of diagnosis, type, prevalence for each patient split, AUC-ROC
test = pd.read_csv(data_path+'test_listfile.csv')
train = pd.read_csv(data_path+'train_listfile.csv')

total_test = test.shape[0]
total_train = train.shape[0]

metrics_df = pd.DataFrame(columns=['Phenotype', 'Type', 'Train', 'Test', 'AUC-ROC'])
i=0
for rx, metrics in final_results_dict.items():
    prev_train = round((train[rx].sum()/total_train), 3)
    prev_test = round((test[rx].sum()/total_test), 3)

    auc_roc = round(metrics['value'], 3)
    rx_type = definitions[rx]['type']

    metrics_df.loc[i] = [rx, rx_type, prev_train, prev_test, auc_roc]
    i += 1


metrics_df.to_csv(data_path+'evaluation/results_table.csv', float_format='%.3f')




