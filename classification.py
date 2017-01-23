#!/usr/bin/env python

from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

import pickle

from spectral_slip_classifier import *
from features import *
import os


class Classification():
    def __init__(self, options, feature_options):

        self.features = Features(feature_options)
        self.classifier_type = options['classifier_type']
        self.training_type = options['training_type']
        self.train_names = options['train_names']
        self.train_trials = options['train_trials']
        self.validation_names = options['validation_names']
        self.validation_trials = options['validation_trials']

        self.training_dictionary = self.get_training_dictionary()
        self.clf = {}

        for key in sorted(self.training_dictionary.keys()):

            if self.classifier_type == 'linear-svm':
                self.clf[key] = svm.SVC(kernel='linear')
            elif self.classifier_type == 'rbf-svm':
                self.clf[key] = svm.SVC(kernel='rbf', gamma=2.0, C=1)
            elif self.classifier_type == 'rdf':
                self.clf[key] = RandomForestClassifier(n_estimators=200, bootstrap=False)
            elif self.classifier_type == 'knn':
                self.clf[key] = neighbors.KNeighborsClassifier(15, weights='uniform')
            elif self.classifier_type == 'ss':
                self.clf[key] = SpectralSlipClassifier()
            elif self.classifier_type == 'gtb':
                self.clf[key] = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=6)
            elif self.classifier_type == 'adaboost':
                self.clf[key] = AdaBoostClassifier(n_estimators=100)
            elif self.classifier_type == 'nbayes':
                self.clf[key] = GaussianNB()
            elif self.classifier_type == 'none':
                self.clf[key] = []

    def train(self):

        for key in sorted(self.training_dictionary):
            print 'Loading Training Data'
            data = self.features.get_data(self.training_dictionary[key][0], self.training_dictionary[key][1])

            print 'Getting Features'
            X, y = self.features.get_features(data)

            print 'Training ' + key + ' Classifier'
            self.clf[key].fit(X, y)

    def validate(self):

        for key in sorted(self.training_dictionary):
            for obj in self.training_dictionary[key][2]:
                for trial in range(self.training_dictionary[key][3][0], self.training_dictionary[key][3][1] + 1):
                    print 'Loading Validation Data'
                    data = self.features.get_data([obj], [trial, trial])

                    print 'Getting Features'
                    X, y = self.features.get_features(data)

                    print 'Predicting Labels'

                    # if self.classifier_type == 'ss':
                    #    Y = []
                    #    label_time, predicted_labels = self.clf[key].predict(np.array(X))

                    #    for i in range(len(label_time)):
                    #        tmp = int(label_time[i] * 100)
                    #        Y.append(y[tmp])

                    # else:


                    predicted_labels = self.clf[key].predict(X)
                    Y = np.array(map(float, y))

                    self.write_evaluation(key, obj, trial, predicted_labels, Y)

    def get_training_dictionary(self):
        training_dictionary = {}
        if self.training_type == 'po':
            for n in self.train_names:
                training_dictionary[n] = [[n], self.train_trials, [n], self.validation_trials]
        elif self.training_type == 'ao':
            training_dictionary[str(self.train_names)] = [self.train_names, self.train_trials, self.validation_names,
                                                          self.validation_trials]
        elif self.training_type == 'loo':
            for i in range(len(self.train_names)):
                nt = self.train_names[:i] + self.train_names[(i + 1):]
                n = self.train_names[i]
                training_dictionary[n] = [nt, self.train_trials, [n], self.validation_trials]

        return training_dictionary

    def write_evaluation(self, key, val, trial, cl, tl):

        training_names = ''
        validation_names = ''
        for c in self.training_dictionary[key][0]:
            training_names += str(c) + ','
        for c in self.training_dictionary[key][2]:
            validation_names += str(c) + ','

        filename = self.classifier_type + '_training_[' + training_names[:-1] + '][' + str(
                self.training_dictionary[key][1][0]) + ',' + str(
                self.training_dictionary[key][1][1]) + ']_validation_[' + val + ']_' + str(trial).zfill(3) + '.results'

        f = open(filename, 'w')

        for i in range(len(cl)):
            f.write(str(cl[i]) + '\t' + str(tl[i]) + '\n')

        f.close()


class OnlineClassification:
    def __init__(self, training_type, classifier_type, train_names, train_trials, target_name, target_trials,
                 feature_type,
                 prediction_type):

        options, feature_options = self.commandline_parser(training_type, classifier_type, train_names, train_trials,
                                                           target_name, target_trials, feature_type,
                                                           prediction_type)

        self.features = Features(feature_options)
        self.classifier_type = options['classifier_type']
        self.training_type = options['experiment_type']
        self.train_names = options['train_names']
        self.train_trials = options['train_trials']
        self.validation_names = options['validation_names']
        self.validation_trials = options['validation_trials']

        self.training_dictionary = self.get_training_dictionary()
        self.clf = {}

        self.last = list(np.zeros(33))

        for key in sorted(self.training_dictionary.keys()):

            if self.classifier_type == 'linear-svm':
                self.clf[key] = svm.SVC(kernel='linear')
            elif self.classifier_type == 'rbf-svm':
                self.clf[key] = svm.SVC(kernel='rbf', gamma=2.0, C=1)
            elif self.classifier_type == 'rdf':
                self.clf[key] = RandomForestClassifier(n_estimators=200, bootstrap=False)
            elif self.classifier_type == 'knn':
                self.clf[key] = neighbors.KNeighborsClassifier(15, weights='uniform')
            elif self.classifier_type == 'ss':
                self.clf[key] = SpectralSlipClassifier()
            elif self.classifier_type == 'gtb':
                self.clf[key] = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=6)
            elif self.classifier_type == 'adaboost':
                self.clf[key] = AdaBoostClassifier(n_estimators=100)
            elif self.classifier_type == 'nbayes':
                self.clf[key] = GaussianNB()
            elif self.classifier_type == 'none':
                self.clf[key] = []

    def train(self):

        cd = '/home/flip/grip_stabilization_ws'

        for key in sorted(self.training_dictionary):
            try:
                print "Attempting to load Classifier\n"
                os.chdir(cd + "/classifiers/")
                pkl_file = open(
                    self.classifier_type + '_' + self.features.feature_type + '_' + self.training_type + '_' + str(
                            self.training_dictionary[key][0]) + '_' + str(self.training_dictionary[key][1]) + '_' +
                    self.features.prediction_type[0] + '_' + str(self.features.prediction_type[1]) + '.pkl', 'rb')
                print "Classifier Found\n"
                clf = pickle.load(pkl_file)
                self.clf[key] = clf
                print "Loading Complete\n"
            except:
                print "Loading Failed, Training New Classifier...\n"
                os.chdir(cd + "/data/")
                print 'Loading Training Data'

                data = self.features.get_data(self.training_dictionary[key][0], self.training_dictionary[key][1])

                print 'Getting Features'
                X, y = self.features.get_features(data)

                print 'Training ' + key + ' Classifier'
                self.clf[key].fit(X, y)
                print 'Saving Classifier'

                os.chdir(cd + "/classifiers/")
                pkl_file = open(
                    self.classifier_type + '_' + self.features.feature_type + '_' + self.training_type + '_' + str(
                            self.training_dictionary[key][0]) + '_' + str(self.training_dictionary[key][1]) + '_' +
                    self.features.prediction_type[0] + '_' + str(self.features.prediction_type[1]) + '.pkl', 'wb')
                pickle.dump(self.clf[key], pkl_file)
                print 'Training Complete'


        os.chdir(cd)

    def classify(self, x):

        predicted_label = 0
        #print 'CLASSIFY'
        ft = self.features.get_online_features(x, self.last)

        for key in sorted(self.training_dictionary):
            predicted_label = self.clf[key].predict([ft])

        self.last = x
        #print 'PYTHON: ' + str(predicted_label)
        return predicted_label

    def validate(self):

        cd = os.getcwd()
        os.chdir(cd + "/data/")

        for key in sorted(self.training_dictionary):
            for obj in self.training_dictionary[key][2]:
                for trial in range(self.training_dictionary[key][3][0], self.training_dictionary[key][3][1] + 1):
                    print 'Loading Validation Data'
                    data = self.features.get_data([obj], [trial, trial])

                    print 'Getting Features'
                    X, y = self.features.get_features(data)

                    print 'Predicting Labels'

                    predicted_labels = self.clf[key].predict(X)
                    Y = np.array(map(float, y))

                    self.write_evaluation(key, obj, trial, predicted_labels, Y)

        os.chdir(cd)

    def get_training_dictionary(self):
        training_dictionary = {}
        if self.training_type == 'po':
            for n in self.train_names:
                training_dictionary[n] = [[n], self.train_trials, [n], self.validation_trials]
        elif self.training_type == 'ao':
            training_dictionary[str(self.train_names)] = [self.train_names, self.train_trials, self.validation_names,
                                                          self.validation_trials]
        elif self.training_type == 'loo':
            for i in range(len(self.train_names)):
                if self.train_names[i] == self.validation_names[0]:
                    index = i
            nt = self.train_names[:index] + self.train_names[(index + 1):]
            n = self.train_names[index]
            training_dictionary[n] = [nt, self.train_trials, [n], self.validation_trials]

        return training_dictionary

    def commandline_parser(self, training_type, classifier_type, train_names, train_trials, target_name, target_trials,
                           feature_type,
                           prediction_type):

        experiment_type = training_type

        classifier_type = classifier_type

        train_names = train_names[1:-1].split(',')
        validation_names = target_name[1:-1].split(',')
        train_trials = map(int, train_trials[1:-1].split(','))
        validation_trials = map(int, target_trials[1:-1].split(','))

        feature_type = feature_type

        prediction_type = prediction_type.split('_')

        prediction_type[1] = int(prediction_type[1])

        options = {'experiment_type': experiment_type,
                   'classifier_type': classifier_type,
                   'train_names': train_names,
                   'validation_names': validation_names,
                   'train_trials': train_trials,
                   'validation_trials': validation_trials}

        feature_options = {'feature_type': feature_type,
                           'prediction_type': prediction_type,
                           'data_type': 'SL_log_SP'}

        return options, feature_options

    def write_evaluation(self, key, val, trial, cl, tl):

        training_names = ''
        validation_names = ''
        for c in self.training_dictionary[key][0]:
            training_names += str(c) + ','
        for c in self.training_dictionary[key][2]:
            validation_names += str(c) + ','

        filename = self.classifier_type + '_training_[' + training_names[:-1] + '][' + str(
                self.training_dictionary[key][1][0]) + ',' + str(
                self.training_dictionary[key][1][1]) + ']_validation_[' + val + ']_' + str(trial).zfill(3) + '.results'

        f = open(filename, 'w')

        for i in range(len(cl)):
            f.write(str(cl[i]) + '\t' + str(tl[i]) + '\n')

        f.close()
