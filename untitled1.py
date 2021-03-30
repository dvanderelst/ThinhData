#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:25:08 2021

@author: dieter
"""

import numpy as np
from matplotlib import pyplot


def ratio2db(ratio):
    db = 20 * np.log10(ratio)
    return db


def db2ratio(db):
    db = np.array(db, dtype='f')
    db = db.astype(float)
    ratio = 10 ** (db / 20.0)
    return ratio

######################################################
#
######################################################

vs = 340
fs = 300000
n = 7000
alpha = -1.3


t_max = n / fs


data10 = np.load('1meter.npy')
data05 = np.load('0.5meter.npy')



time_axis = np.linspace(0,t_max, n)
travel_distance_axis = time_axis * vs * 2

start_distance = 1
target_distance = 0.5

travel_distance_shift = 2 * (target_distance - start_distance)

# samples to shift
samples_shift = travel_distance_shift * fs / vs 
samples_shift = np.round(samples_shift)
samples_shift = int(samples_shift)

# atmospheric attenuation
attenuation_db = alpha * travel_distance_shift
attenuation_linear = db2ratio(attenuation_db)

# spreading attenuation
new_travel_distance_axis = travel_distance_axis + travel_distance_shift

travel_distance_axis_truncated = travel_distance_axis
new_travel_distance_axis_truncated = new_travel_distance_axis
travel_distance_axis_truncated[travel_distance_axis_truncated<0.1] = 0.1
new_travel_distance_axis_truncated[new_travel_distance_axis_truncated<0.1] = 0.1

spreading_loss = (1/travel_distance_axis_truncated)
new_spreading_loss =  (1/new_travel_distance_axis_truncated)


spreading_loss_ratio = new_spreading_loss/spreading_loss


## Apply
new_waveform = np.roll(data10, shift=samples_shift)
new_waveform = new_waveform * attenuation_linear
new_waveform = new_waveform * spreading_loss_ratio


pyplot.plot(travel_distance_axis, data05, alpha=0.5)
pyplot.plot(travel_distance_axis, data10, alpha=0.5)
pyplot.legend(['0.5', '1.0'])

pyplot.figure()

pyplot.plot(travel_distance_axis, data05, alpha=0.5)
pyplot.plot(travel_distance_axis, new_waveform, alpha=0.5)
pyplot.legend(['0.5', 'new'])
