import pandas as pd

'''
returns total number of patients per split
'''

data_path = '../data/phenotyping/'


def get_patient_stats(file):
    data = pd.read_csv(file)
    return data.shape[0]


train = data_path + 'train_listfile.csv'
test = data_path + 'test_listfile.csv'
val = data_path + 'val_listfile.csv'

print('train nr. patients', get_patient_stats(train))
print('test nr. patients', get_patient_stats(test))
print('val nr. patients', get_patient_stats(val))