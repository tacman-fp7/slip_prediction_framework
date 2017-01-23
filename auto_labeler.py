#!/usr/bin/env python

import numpy as np
from load_data import *


def auto_labeler(data):

    contact_threshold = 8
    motion_threshold = 0.00005
    labels = [0]

    sample_length = len(data['pdc'])

    for i in range(1, sample_length):

        if data['pdc'][i] > contact_threshold:
            motion = np.sqrt(pow(data['cart_x'][i]-data['cart_x'][i-1], 2) + pow(data['cart_y'][i]-data['cart_y'][i-1], 2) + pow(data['cart_z'][i] - data['cart_z'][i-1], 2))

            if motion > motion_threshold:
                labels.append(2)
            else:
                labels.append(1)

        else:
            labels.append(0)

    data['labels'] = labels

    return data
