########################
# Plot a basic figure. #
########################


import numpy as np
import matplotlib.pyplot as plt


thresholds = np.array(  [  0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,   ]  )
precisions = np.array(  [  79.91,  84.06,  86.33,  88.08,  89.31,  90.50,  92.35,  93.39,  92.50  ]  )
recalls    = np.array(  [  96.63,  95.18,  92.92,  91.02,  89.18,  87.78,  83.27,  78.35,  70.91  ]  )
f_measures = np.array(  [  87.48,  89.28,  89.50,  89.53,  89.24,  89.12,  87.58,  85.21,  80.28  ]  )


# a = plt.plot( thresholds, precisions, 'm^', label='_nolegend_' )
# b = plt.plot( thresholds, precisions, 'm',  label='Precision'  )
# c = plt.plot( thresholds, recalls,    'ro', label='_nolegend_' )
# d = plt.plot( thresholds, recalls,    'r',  label='Recall   '  )
# e = plt.plot( thresholds, f_measures, 'b*', label='_nolegend_' )
# f = plt.plot( thresholds, f_measures, 'b',  label='F-score'    )


a = plt.plot( thresholds, precisions, color='m', marker='^', label='Precision' )
b = plt.plot( thresholds, recalls,    color='r', marker='o', label='Recall   ' )
c = plt.plot( thresholds, f_measures, color='b', marker='*', label='F-score'   )


plt.xlabel('Sigmoid Threshold')
plt.ylabel('Metric Percentage')
plt.legend()


for i,j in zip(thresholds, f_measures):

    if i == 0.3:
        plt.text( i - 0.037, j - 1.19, str(j) )

    elif i == 0.4:
        plt.text( i - 0.05,  j - 1.2,  str(j) )

    elif i == 0.6:
        plt.text( i - 0.015, j - 1.1,  str(j) )

    elif i == 0.7:
        plt.text( i - 0.04,  j - 1.6,  str(j) )

    elif i == 0.8:
        plt.text( i - 0.055, j - 1.6,  str(j) )

    else:
        plt.text( i - 0.034, j - 1.6,  str(j) )


plt.savefig('thresh.eps', bbox_inches='tight', format='eps', dpi=300)
plt.show()
