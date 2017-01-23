import numpy as np
from load_data import *
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis

import warnings

warnings.simplefilter('ignore', np.RankWarning)


class Features():
    def __init__(self, options):
        self.feature_type = options['feature_type']
        self.prediction_type = options['prediction_type']
        self.data_type = options['data_type']

    def get_data(self, name, tset):
        f_type = self.feature_type.split('_')
        data = load_data(name, self.data_type, len(tset), tset, f_type[0])

        return data

    def get_features(self, data):

        X = []
        y = []

        f_type = self.feature_type.split('_')

        if f_type[0] == 'raw':
            for i in range(len(data['labels']) - self.prediction_type[1]):
                # if data['labels'][i + self.prediction_type[1]] != 0:
                if len(f_type) > 1:
                    if f_type[1] == 'pdc':
                        X.append(list([data['pdc'][i]]))
                    elif f_type[1] == 'pac':
                        X.append(data['pac'][i])
                    elif f_type[1] == 'electrode':
                        X.append(data['electrode'][i])
                    elif f_type[1] == 'tdc':
                        X.append(list([data['tdc'][i]]))
                    elif f_type[1] == 'tac':
                        X.append(list([data['tac'][i]]))
                else:
                    X.append(list([data['pdc'][i]]) + data['pac'][i] + data['electrode'][i] + list(
                            [data['tdc'][i]]) + list([data['tac'][i]]))
                if self.prediction_type[1] == 0:
                    y.append(data['labels'][i])
                else:
                    y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'delta':
            for i in range(1, len(data['labels']) - self.prediction_type[1]):
                # if data['labels'][i + self.prediction_type[1]] != 0:
                delta = list([data['pdc'][i] - data['pdc'][i - 1]]) + list(
                        np.array(data['pac'][i]) - np.array(data['pac'][i - 1])) + list(
                        np.array(data['electrode'][i]) - np.array(data['electrode'][i - 1])) + list(
                        [data['tdc'][i] - data['tdc'][i - 1]]) + list([data['tac'][i] - data['tac'][i - 1]])
                X.append(list([data['pdc'][i]]) + data['pac'][i] + data['electrode'][i] + list(
                        [data['tdc'][i]]) + list([data['tac'][i]]) + delta)
                if self.prediction_type[1] == 0:
                    y.append(data['labels'][i])
                else:
                    y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'deltanoPAC':
            for i in range(1, len(data['labels']) - self.prediction_type[1]):
                if data['labels'][i + self.prediction_type[1]] != 0:
                    delta = list([data['pdc'][i] - data['pdc'][i - 1]]) + list(
                            np.array(data['electrode'][i]) - np.array(data['electrode'][i - 1])) + list(
                            [data['tdc'][i] - data['tdc'][i - 1]]) + list([data['tac'][i] - data['tac'][i - 1]])
                    X.append(list([data['pdc'][i]]) + data['pac'][i] + data['electrode'][i] + list(
                            [data['tdc'][i]]) + list([data['tac'][i]]) + delta)
                    if self.prediction_type[1] == 0:
                        y.append(data['labels'][i])
                    else:
                        y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'deltaPACfix':
            for i in range(1, len(data['labels']) - self.prediction_type[1]):
                if data['labels'][i + self.prediction_type[1]] != 0:
                    delta_pac = [data['pac'][i][0] - data['pac'][i - 1][0]]
                    for j in range(1, len(data['pac'][i])):
                        delta_pac.append(data['pac'][i][j] - data['pac'][i][j - 1])

                    delta = list([data['pdc'][i] - data['pdc'][i - 1]]) + delta_pac + list(
                            np.array(data['electrode'][i]) - np.array(data['electrode'][i - 1])) + list(
                            [data['tdc'][i] - data['tdc'][i - 1]]) + list([data['tac'][i] - data['tac'][i - 1]])
                    X.append(list([data['pdc'][i]]) + data['pac'][i] + data['electrode'][i] + list(
                            [data['tdc'][i]]) + list([data['tac'][i]]) + delta)
                    if self.prediction_type[1] == 0:
                        y.append(data['labels'][i])
                    else:
                        y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'tw':
            w_size = int(f_type[1])
            for i in range(w_size, len(data['labels']) - self.prediction_type[1]):
                if data['labels'][i + self.prediction_type[1]] != 0:
                    window = []
                    for j in range(i - w_size, i + 1):
                        window.extend(list([data['pdc'][j]]) + data['pac'][j] + data['electrode'][j])
                    X.append(window)
                    if self.prediction_type[1] == 0:
                        y.append(data['labels'][i])
                    else:
                        y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'convtest':
            w_size = int(f_type[1])
            for i in range(w_size, len(data['labels']) - self.prediction_type[1]):
                if data['labels'][i + self.prediction_type[1]] != 0:
                    window = []
                    for k in range(19):
                        window_elec = []
                        window_pac = []
                        for j in range(i - w_size, i + 1):
                            window_elec.append(data['electrode'][j][k])
                            window_pac.extend(data['pac'][j] - np.mean(data['pac'][j]))
                        window.append(np.convolve(window_elec, window_pac))
                    if X == []:
                        X = window
                    else:
                        for k in range(19):
                            X[k] = np.concatenate((X[k], window[k]), 0)
                    if self.prediction_type[1] == 0:
                        y.extend(np.ones(len(window[0])) * data['labels'][i])
                    else:
                        y.extend(np.ones(len(window[0])) * data['labels'][i + self.prediction_type[1]])
            Xaux = []
            for i in range(len(X[0])):
                fet = []
                for k in range(19):
                    fet.append(X[k][i])
                Xaux.append(fet)
            X = Xaux

        elif f_type[0] == 'ss':
            for i in range(len(data['labels']) - self.prediction_type[1]):
                if data['labels'][i + self.prediction_type[1]] != 0:
                    X.append(data['pac'][i])
                    if self.prediction_type[1] == 0:
                        y.append(data['labels'][i])
                    else:
                        y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'chu':
            X, y = self.get_chu_features(data)

        elif f_type[0] == 'rawWchuPAC':
            for i in range(len(data['labels']) - self.prediction_type[1]):
                # if data['labels'][i + self.prediction_type[1]] != 0:
                X.append(
                    list([data['pdc'][i]]) + self.get_chu_roughness(data['pac'][i], multiple=False) + data['electrode'][
                        i] + list(
                            [data['tdc'][i]]) + list([data['tac'][i]]))
                if self.prediction_type[1] == 0:
                    y.append(data['labels'][i])
                else:
                    y.append(data['labels'][i + self.prediction_type[1]])

        elif f_type[0] == 'deltaWchuPAC':
            for i in range(1, len(data['labels']) - self.prediction_type[1]):
                # if data['labels'][i + self.prediction_type[1]] != 0:
                delta = list([data['pdc'][i] - data['pdc'][i - 1]]) + list(
                    np.array(self.get_chu_roughness(data['pac'][i], multiple=False)) - np.array(
                        self.get_chu_roughness(data['pac'][i - 1], multiple=False))) + list(
                        np.array(data['electrode'][i]) - np.array(data['electrode'][i - 1])) + list(
                        [data['tdc'][i] - data['tdc'][i - 1]]) + list([data['tac'][i] - data['tac'][i - 1]])
                X.append(
                    list([data['pdc'][i]]) + self.get_chu_roughness(data['pac'][i], multiple=False) + data['electrode'][
                        i] + list(
                            [data['tdc'][i]]) + list([data['tac'][i]]) + delta)
                if self.prediction_type[1] == 0:
                    y.append(data['labels'][i])
                else:
                    y.append(data['labels'][i + self.prediction_type[1]])
        return X, y

    def get_online_features(self, X, last):

        if self.feature_type == 'ss':
            return [X[1:23]]

        elif self.feature_type == 'raw':
            return X

        elif self.feature_type == 'delta':
            delta = list(np.array(X) - np.array(last))
            feat = X + delta
            return feat

        elif self.feature_type == 'rawWchuPAC':
            feat = [X[0]] + self.get_chu_roughness(X[1:8], multiple=False) + X[23:]
            return feat

        elif self.feature_type == 'deltaWchuPAC':
            delta = [X[0] - last[0]] + list(np.array(self.get_chu_roughness(X[1:8], multiple=False)) - np.array(
                self.get_chu_roughness(last[1:8], multiple=False))) + list(np.array(X[8:33]) - np.array(last[8:33]))
            feat = [X[0]] + self.get_chu_roughness(X[1:8], multiple=False) + X[8:] + delta
            return feat

    def get_chu_features(self, data):
        X = []
        y = []
        f_type = self.feature_type.split('_')

        w_size = int(f_type[1])

        for i in range(w_size, len(data['labels']) - self.prediction_type[1]):
            if data['labels'][i] != 0:
                electrode_data = []
                pac_data = []

                for j in range(i - w_size, i):
                    electrode_data.append(data['electrode'][j])
                    pac_data.append(data['pac'][j])
                tdc_data = data['tdc'][i - w_size:i]
                tac_data = data['tac'][i - w_size:i]
                pdc_data = data['pdc'][i - w_size:i]

                X.append(self.get_chu_electrodes(electrode_data) + self.get_chu_thermal(tdc_data,
                                                                                        tac_data) + self.get_chu_roughness(
                    pac_data) + self.get_chu_compliance(pdc_data))
                if self.prediction_type[1] == 0:
                    y.append(data['labels'][i])
                else:
                    y.append(data['labels'][i + self.prediction_type[1]])

        return X, y

    def get_chu_electrodes(self, electrode_data):

        pca = PCA(n_components=2)
        pca.fit(electrode_data)
        electrode_data_pc = pca.transform(electrode_data)

        p1 = np.polyfit(range(len(electrode_data_pc[:, 0])), electrode_data_pc[:, 0], 5)
        p2 = np.polyfit(range(len(electrode_data_pc[:, 1])), electrode_data_pc[:, 1], 5)

        p = list(np.concatenate((p1, p2), 0))

        return p

    def get_chu_thermal(self, tdc, tac):

        r1 = np.trapz(tac)
        x = range(1, len(tdc) + 1)

        r2 = np.linalg.lstsq(np.mat(x).T, np.log(np.mat(tdc) + abs(min(np.mat(tdc))) + 0.00001).T)[0]

        r = [r1, np.array(r2)[0][0]]

        return r

    def get_chu_roughness(self, pac, multiple=True):

        if multiple:
            PAC = []

            for p in pac:
                PAC = PAC + p
        else:
            PAC = pac

        esd = np.power(abs(np.fft.fft(PAC)), 2)

        results = [np.trapz(esd), np.mean(esd), np.var(esd), skew(esd), kurtosis(esd)]

        return results

    def get_chu_compliance(self, pdc):

        maxChange = 0
        for i in range(len(pdc) - 1):
            if abs(pdc[i] - pdc[i + 1]) > maxChange:
                maxChange = abs(pdc[i] - pdc[i + 1])

        results = [max(pdc), np.mean(pdc), maxChange]

        return results
