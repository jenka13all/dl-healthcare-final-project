import pandas as pd
import matplotlib.pyplot as plt

data_path = '../data/phenotyping/'
train = pd.read_csv(data_path + 'train_listfile.csv')
test = pd.read_csv(data_path + 'test_listfile.csv')
data = pd.concat([train, test])

data[['pat_id', 'visit', 'filename']] = data['stay'].str.split('_', expand=True)
data = data[['pat_id', 'visit']]

pat_visits = data.groupby('pat_id').count().reset_index()

# will call the count column "pat_id" - it's really a count of pat_ids
visit_pats = pat_visits.groupby('visit').count().reset_index()

# histogram of visit number distribution
plt.bar(list(visit_pats.loc[0:4, 'visit']), list(visit_pats.loc[0:4, 'pat_id']), color='g')
plt.title('Patient Visits Distribution')
plt.xlabel('Number of visits')
plt.ylabel('Number of patients with that many visits')
plt.savefig('../figures/visit_distribution.png', dpi=700)

# print some basic statistics about the variance of the number of visits
nr_visits = visit_pats['visit'].tolist()
print('maximum number of visits for a patient', max(nr_visits))
print('minimum number of visits for a patient', min(nr_visits))
