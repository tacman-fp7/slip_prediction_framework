#!/usr/bin/env python

from spectral_slip import *
import matplotlib.pyplot as plt

class SpectralSlipClassifier():
    def __init__(self, threshold=100000.0):
        self.threshold = threshold

    def fit(self, X, y):

        labels = []

        for f in X:
            clf_time, clf_label = compute_fishel_spectral_slip(np.array(f))
            labels.append(abs(clf_label))

        c_score = 0
        for t in np.arange(1e7, 7e7, 5e5):
            score = 0
            label = []
            for i in range(len(labels)):
                if labels[i] >= t:
                    label.append(2)
                else:
                    label.append(1)

            for j in range(len(label)):
                if label[j] == y[j]:
                    score += 1

            if score > c_score:
                self.threshold = t
                c_score = score

    def predict(self, X):

        labels = []
        for f in X:
            clf_time, clf_label = compute_fishel_spectral_slip(np.array(f))
            labels.append(clf_label)

        for i in range(len(labels)):
            if labels[i] >= self.threshold:
                labels[i] = 2
            else:
                labels[i] = 1

	if(len(labels) == 1):
	    return labels[0]

        return labels
