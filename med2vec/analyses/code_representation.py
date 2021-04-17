import numpy as np
from matplotlib import pyplot as plt

weights = np.load('../models/models.0.npz')
model_dict = dict(zip(("{}".format(k) for k in weights), (weights[k] for k in weights)))

#print(model_dict.keys())
#['W_emb', 'b_output', 'b_hidden', 'b_emb', 'W_output', 'W_hidden']

three_digit_icd9_dict = {
    'D_E900': 924, 'D_E861': 865,
    'D_624': 892, 'D_E919': 751,
    'D_838': 809, 'D_854': 887,
    'D_853': 79, 'D_852': 80,
    'D_998': 68, 'D_999': 20
}

labels, disease_list = zip(*three_digit_icd9_dict.items())
labels = list(labels)
disease_list = list(disease_list)

W_emb = model_dict['W_emb']

plt.scatter(W_emb[disease_list, 0], W_emb[disease_list, 1], s=10)
for label, x, y in zip(labels, W_emb[disease_list, 0], W_emb[disease_list, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.savefig('figures/code_show.png', dpi=700)