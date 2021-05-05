import pandas as pd
import os

'''
select patients from phenotyping task patient dataset that have the presence
of at least one of the care conditions in order to clustering their learned representations. 
'''
pheno_data_path = '../../../mimic3-benchmarks/data/phenotyping/'
train_listfile = pd.read_csv(os.path.join(pheno_data_path, 'train', 'listfile.csv'))
test_listfile = pd.read_csv(os.path.join(pheno_data_path, 'test', 'listfile.csv'))


def get_subject_ids(df):
    diag_only = df.iloc[:, 2:27]
    pheno_pos = df[diag_only.sum(axis=1) > 1]
    pheno_pos[['subject_id', 'stay_file']] = pheno_pos['stay'].str.split('_', n=1, expand=True)
    subject_ids = list(pheno_pos['subject_id'])

    return subject_ids


train_subject_ids = get_subject_ids(train_listfile)
test_subject_ids = get_subject_ids(test_listfile)

print('train subjects total:', train_listfile.shape[0])
print('train subjects with condition:', len(train_subject_ids))
print('test subjects total:', test_listfile.shape[0])
print('test subjects with condition:', len(test_subject_ids))

# test set, need test_subject_ids
# trained_vectors = pd.read_csv('../data/encodings/convae_vect.csv')
