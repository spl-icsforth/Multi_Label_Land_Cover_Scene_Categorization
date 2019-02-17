##########################
# Plot a basic bar plot. #
##########################


import numpy as np
import matplotlib.pyplot as plt


N = 3

# Order --- 400, 800, 1600
no_augment  = (71.43, 78.09, 83.09)
augment     = (81.16, 86.17, 88.63)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig1, ax1 = plt.subplots()

rects1 = ax1.bar(ind, no_augment, width, color='r')

rects2 = ax1.bar(ind + width, augment, width, color='g')

ax1.set_ylabel('F-score (%)')
ax1.set_xlabel('Initial Training Set Size')
ax1.set_xticks(ind + width / 2)
ax1.set_xticklabels(('400 Samples', '800 Samples', '1600 Samples'))
ax1.set_ylim(50, 100)

ax1.legend((rects1[0], rects2[0]), ('No Augmentation', 'With Augmentation'), loc=4)

def autolabel(rects,ax):

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.015*height,
                '%.2f' % height,
                ha='center', va='bottom')

autolabel(rects1,ax1)
autolabel(rects2,ax1)

plt.savefig('tr_samples.eps', bbox_inches='tight', format='eps', dpi=300)
plt.show()