#!/usr/bin/env python


import struct
import numpy as np
import auto_labeler as al

fingers = ['th', 'in', 'mi']

def load_data(names, data_type, nfiles=0, interval=[], Ftype='raw', exp_type=['h'], pressure_v=[20, 30, 40, 50]):
    #print data_type
    if data_type == 'SL_log':
        data = read_data_SL(names, interval, exp_type, pressure_v)
    elif data_type == 'SL_log_SP':
        data = read_data_SL_SP(names, interval, exp_type, pressure_v)
    else:
        data = read_data(names, nfiles, interval, Ftype)

    return data


def read_data(names, nfiles, interval, Ftype):
    if Ftype in ['raw', 'delta', 'tw', 'ss']:
        tdc = []
        tac = []
        pdc = []
        pac = []
        electrode = []
        labels = []
    else:
        compliance = []
        roughness = []
        thermal = []
        electrode = []
        labels = []

    for name in names:

        if nfiles == 0:
            filename = [name + '.data']
        elif nfiles == 1:
            filename = [name + str(interval[0]).zfill(3) + '.data']
        else:
            filename = []
            if interval == []:
                for i in range(nfiles):
                    filename.append(name + str(i + 1).zfill(3) + '.data')
            else:
                for i in range(interval[0], interval[1] + 1):
                    filename.append(name + str(i).zfill(3) + '.data')

        for fname in filename:

            f = open(fname, 'r')

            for line in f:
                if Ftype in ['raw', 'delta', 'tw', 'ss']:
                    s = map(int, line.split())

                    electrode.append(list(s[0:19]))
                    pac.append(list(s[19:41]))
                    pdc.append(s[41])
                    tac.append(s[42])
                    tdc.append(s[43])
                    labels.append(s[44])
                else:
                    s = map(float, line.split())

                    compliance.append(list(s[0:3]))
                    roughness.append(list(s[3:8]))
                    thermal.append(list(s[8:11]))
                    electrode.append(list(s[11:22]))
                    labels.append(s[22])

            f.close
    if Ftype in ['raw', 'delta', 'tw', 'ss']:
        data = {'tdc': tdc,
                'tac': tac,
                'pdc': pdc,
                'pac': pac,
                'electrode': electrode,
                'labels': labels}
    else:
        data = {'compliance': compliance,
                'roughness': roughness,
                'thermal': thermal,
                'electrode': electrode,
                'labels': labels}
    return data


def read_data_SL(names, interval, exp_type, pressure_v):
    tdc = []
    tac = []
    pdc = []
    pac = []
    electrode = []
    cart_x = []
    cart_y = []
    cart_z = []
    labels = []
    filename = []

    for name in names:
        for e_type in exp_type:
            for p_v in pressure_v:
                for i in range(interval[0], interval[1]+1):
                    p_name = name + '_' + e_type + '_' + str(p_v) + '_' + str(i).zfill(3)
                    #print p_name
                    filename = filename + get_SL_file_list_from_key('log_file_key.txt', p_name)
                    #print filename
                    #raw_input()

    for fname in filename:
        file_data = load_sl_data(fname)
        file_data = al.auto_labeler(file_data)
        electrode += file_data['electrode']
        pac += file_data['pac']
        pdc += file_data['pdc']
        tac += file_data['tac']
        tdc += file_data['tdc']
        cart_x += file_data['cart_x']
        cart_y += file_data['cart_y']
        cart_z += file_data['cart_z']
        labels += file_data['labels']
        #print 'ELEC'
        #print electrode
        #print 'PAC'
        #print pac
        #print 'PDC'
        #print pdc
        #raw_input()

    data = {'tdc': tdc,
            'tac': tac,
            'pdc': pdc,
            'pac': pac,
            'electrode': electrode,
            'labels': labels}

    return data


def read_data_SL_SP(names, interval, exp_type, pressure_v):
    tdc = []
    tac = []
    pdc = []
    pac = []
    electrode = []
    cart_x = []
    cart_y = []
    cart_z = []
    labels = []
    filename = []

    for name in names:
        for e_type in exp_type:
            for p_v in pressure_v:
                for i in range(interval[0], interval[1] + 1):
                    p_name = name + '_' + e_type + '_' + str(p_v) + '_' + str(i).zfill(2)
                    #print p_name
                    # print p_name
                    filename = filename + get_SL_file_list_from_key('log_file_key.txt', p_name)
                    # print filename
                    # raw_input()

                    # print filename
    iter = 0
    n_f = len(filename)
    #print filename
    for fname in filename:
        print 'Loading file number ' + str(iter) + ' out of ' + str(n_f) + ' files'
        file_data = load_sl_data_sp(fname)

        file_data = al.auto_labeler(file_data)

        electrode += file_data['electrode']
        pac += file_data['pac']
        pdc += file_data['pdc']
        tac += file_data['tac']
        tdc += file_data['tdc']
        cart_x += file_data['cart_x']
        cart_y += file_data['cart_y']
        cart_z += file_data['cart_z']
        labels += file_data['labels']
        iter += 1

        # print 'ELEC'
        # print electrode
        # print 'PAC'
        # print pac
        # print 'PDC'
        # print pdc
        # raw_input()


    data = {'tdc': tdc,
            'tac': tac,
            'pdc': pdc,
            'pac': pac,
            'electrode': electrode,
            'labels': labels}

    return data


def read_SL_log(file_name):
    '''
    Read in an SL log file
    file_name - path to the SL data file to read (d0*)
    return - (data_set, var_names, var_units)
    data_set - a dictionary of the data with keys as saved variable names
    var_name - a list of variable names in order that they were saved in the log
    var_units - the units associated with the data of var_names
    '''
    data_in = file(file_name, 'rb')
    # Get numerical header
    [numvars, cols, rows, freq] = [float(x) for x in data_in.readline().split()]
    numvars = int(numvars)
    cols = int(cols)
    rows = int(rows)
    # Get variable names and units
    header_line = data_in.readline().split()
    i = 0
    var_names = []
    var_units = []
    while i < len(header_line):
        var_names.append(header_line[i])
        var_units.append(header_line[i + 1])
        i += 2
    data_set = {}
    for k in var_names:
        data_set[k] = []

    # Read in the rest of the binary data
    # Big endian floats
    raw_floats = struct.unpack('>' + 'f' * numvars, data_in.read(4 * numvars))
    data_in.close()
    # pack into the correct dictionary array
    for r in xrange(rows):
        for c in xrange(cols):
            idx = r * cols + c
            data_set[var_names[c]].append(raw_floats[idx])

    # Remove all 0 data
    try:
        N = data_set['time'].index(0, 1)
        for k in data_set.keys():
            data_set[k] = data_set[k][:N]
    except ValueError:
        pass

    for key in data_set.keys():
        data_set[key] = np.array(data_set[key])

    return data_set, var_names, var_units


def load_sl_data(filename):
    v, n, u = read_SL_log(filename)

    pdc = list(v['bt_PDC'] - v['PDC_tare'][0])

    pac = read_pac(v)
    tdc = list(v['bt_TDC'])
    tac = list(v['bt_TAC'])
    electrode = read_electrodes(v)
    cart_x = list(v['R_HAND_x'])
    cart_y = list(v['R_HAND_y'])
    cart_z = list(v['R_HAND_z'])

    data = {'tdc': tdc,
            'tac': tac,
            'pdc': pdc,
            'pac': pac,
            'electrode': electrode,
            'cart_x': cart_x,
            'cart_y': cart_y,
            'cart_z': cart_z}

    return data


def load_sl_data_sp(filename):
    v, n, u = read_SL_log(filename)

    pdc = []
    tdc = []
    tac = []
    cart_x = []
    cart_y = []
    cart_z = []

    for i in range(len(fingers)):
        pdc = pdc + list(v['bt_' + fingers[i] + '_PDC'] - v['pdc_tare_' + str(i)][0])
        tdc = tdc + list(v['bt_' + fingers[i] + '_TDC'])
        tac = tac + list(v['bt_' + fingers[i] + '_TAC'])
        cart_x = cart_x + list(v['R_' + fingers[i].upper() + '_EE_x'])
        cart_y = cart_y + list(v['R_' + fingers[i].upper() + '_EE_y'])
        cart_z = cart_z + list(v['R_' + fingers[i].upper() + '_EE_z'])

    pac = read_pac_sp(v)

    electrode = read_electrodes_sp(v)

    data = {'tdc': tdc,
            'tac': tac,
            'pdc': pdc,
            'pac': pac,
            'electrode': electrode,
            'cart_x': cart_x,
            'cart_y': cart_y,
            'cart_z': cart_z}

    return data


def read_pac(data):
    start_pac = 1
    num_pac = 22
    sample_length = len(data['time'])
    bt_PAC = []

    for j in xrange(sample_length):
        bt_PAC_iter = []
        for i in range(start_pac, start_pac + num_pac):
            bt_idx = str('bt_PAC%.2d' % i)
            bt_PAC_iter.append(data[bt_idx][j])
        bt_PAC.append(bt_PAC_iter)

    return bt_PAC


def read_pac_sp(data):
    start_pac = 1
    num_pac = 6
    sample_length = len(data['time'])
    bt_PAC = []

    for k in range(len(fingers)):
        for j in xrange(sample_length):
            bt_PAC_iter = []
            for i in range(start_pac, start_pac + num_pac):
                bt_idx = 'bt_' + fingers[k] + '_PAC' + str('%.2d' % i)
                bt_PAC_iter.append(data[bt_idx][j])
            bt_PAC.append(bt_PAC_iter)

    return bt_PAC


def read_electrodes(data):
    start_elec = 1
    num_elec = 19
    sample_length = len(data['time'])
    bt_elec = []

    elec_tare = read_electrodes_tare(data)

    for j in xrange(sample_length):
        bt_elec_iter = []
        for i in range(start_elec, start_elec + num_elec):
            bt_idx = str('bt_E%.2d' % i)
            bt_elec_iter.append(data[bt_idx][j] - elec_tare[i-1])

        bt_elec.append(bt_elec_iter)

    return bt_elec


def read_electrodes_sp(data):
    start_elec = 1
    num_elec = 24
    sample_length = len(data['time'])
    bt_elec = []

    for k in range(len(fingers)):
        elec_tare = read_electrodes_tare_sp(data, k)
        for j in xrange(sample_length):
            bt_elec_iter = []
            for i in range(start_elec, start_elec + num_elec):
                bt_idx = 'bt_' + fingers[k] + str('_E%.2d' % i)
                bt_elec_iter.append(data[bt_idx][j] - elec_tare[k*24+i-1])

            bt_elec.append(bt_elec_iter)

    return bt_elec


def read_electrodes_tare(data):
    num_elec = 19
    elec_tare = []

    for i in range(num_elec):
        bt_idx = str('elec_tare_%d' % i)
        elec_tare.append(data[bt_idx][0])

    return elec_tare


def read_electrodes_tare_sp(data, n_finger):
    num_elec = 24
    elec_tare = []

    for k in range(len(fingers)):
        for i in range(num_elec):
            bt_idx = 'E_' + fingers[k] + str('_tare_%d' % i)
            elec_tare.append(data[bt_idx][0])

    return elec_tare


def read_pose(data):
    x = data['R_HAND_x']
    y = data['R_HAND_y']
    z = data['R_HAND_z']
    q0 = data['R_HAND_q0']
    q1 = data['R_HAND_q1']
    q2 = data['R_HAND_q2']
    q3 = data['R_HAND_q3']

    pose = {'cart_x': x,
            'cart_y': y,
            'cart_z': z,
            'q0': q0,
            'q1': q1,
            'q2': q2,
            'q3': q3}

    return pose


def get_SL_file_list_from_key(log_key_file_name, key_name):
    '''
    Method to get names of SL logs associated with a specific key
    log_key_file_name - path to file containing set of (SL_log_file, key) pairs
    key_name - the desired key to use in selecting SL log files
    '''
    sl_base_path = log_key_file_name[:-len(log_key_file_name.split('/')[-1])]
    key_file = file(log_key_file_name, 'r')
    key_pairs = [line.rstrip().split() for line in key_file.readlines()]
    key_file.close()

    sl_file_list = []
    for p in key_pairs:
        if key_name in p[0]:
            # print sl_base_path+p[1]
            sl_file_list.append(sl_base_path + p[1])
    return sl_file_list
