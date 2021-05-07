import pandas as pd

'''
returns various statistic about the data
'''


# get number of visits in a split
def get_visit_stats(filename):
    data = pd.read_csv(filename)
    return data.shape[0]


# get number of patients in a split
def get_patient_stats(filename):
    data = pd.read_csv(filename)
    data_pats = set(data['stay'].str.split('_', n=1, expand=True)[0].tolist())

    return len(data_pats)


def get_total_visits(data_path):
    train_visits = get_visit_stats(data_path + 'train-listfile.csv')
    test_visits = get_visit_stats(data_path + 'test-listfile.csv')

    return train_visits + test_visits


def get_total_pats(data_path):
    train_pats = get_patient_stats(data_path + 'train-listfile.csv')
    test_pats = get_patient_stats(data_path + 'test-listfile.csv')

    return train_pats + test_pats


data_path = '../pat_data/'
print('train visits', get_visit_stats(data_path + 'train-listfile.csv'))
print('train patients', get_patient_stats(data_path + 'train-listfile.csv'))

print('test visits', get_visit_stats(data_path + 'test-listfile.csv'))
print('test patients', get_patient_stats(data_path + 'test-listfile.csv'))

print('total visits', get_total_visits(data_path))
print('total patients', get_total_pats(data_path))
