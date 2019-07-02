# Deep Learning for Multi-label Land Cover Scene Categorization Using Data Augmentation

This repository contains Python code designed for the the problem of multi-label land cover scene categorization. A deep learning architecture is adopted, namely Convolutional Neural Networks, along with the utilization of a data augmentation technique for the artificial increase of the size of the employed dataset.

## Requirements

### Dataset
The performance of the proposed scheme is evaluated on the UC Merced Land Use dataset. Given that UC-Merced is a dataset that originally corresponds to the multi-class classification problem, in this work we adopt a multi-label variation of the dataset. UC-Merced is available in http://weegee.vision.ucmerced.edu/datasets/landuse.html. The modified labelset that we utilize in this work is available in http://bigearth.eu/datasets.html (in the context of the BigEarth research project).

### Framework 
We use the well-known deep learning frameworks, Tensorflow (1.1) and Keras (2.1.3), in Python 3.6.

Tensorflow --> https://www.tensorflow.org/ <br />
Keras --> https://keras.io/ <br />
Python (Anaconda distribution) --> https://www.anaconda.com/ 

## Contents
**load_data.py**: Loads the data that we downloaded from the links in the section "**Dataset**" and transforms them in a suitable (numpy) format.

**cnn_big_earth_keras.py**: Performs the training and testing of the proposed methodology.

**plot_shapes_annotations.py**: Precision, Recall and F-Score plot for the various sigmoid thresholds.

**plot_train_bar_plots.py**: Different bar plots, based on the initial size of the utilized training set (with or without data augmentation).

## References
1. R. Stivaktakis, G. Tsagkatakis, and P. Tsakalides,
“Deep Learning for Multi-label Land Cover Scene Categorization Using Data Augmentation” IEEE Geoscience and Remote Sensing Letters, vol. 16 issue 7, pp. 1031 - 1035, 2019.

2. B. Chaudhuri, B. Demir, S. Chaudhuri, L. Bruzzone,
"Multi-label Remote Sensing Image Retrieval using a Semi-Supervised Graph-Theoretic Method", IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no.1, 2018
