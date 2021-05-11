import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Total', 'Train', 'Test']
bench = [28543, 23466, 5068]
convae = [33043, 28081, 4962]
med2vec = [7537, 6029, 1508]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x + 0.00, bench, width, label='Benchmark')
rects2 = ax.bar(x + 0.25, convae, width, label='ConvAE')
rects3 = ax.bar(x + 0.50, med2vec, width, label='Med2Vec')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Patients')
ax.set_title('Patients by study and split (Total/Train/Test)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig('patients_by_study_and_split.png', dpi=700)
