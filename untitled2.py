#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:25:08 2021

@author: dieter
"""

import numpy as np
from matplotlib import pyplot
from scipy.signal import butter, lfilter


def ratio2db(ratio):
    db = 20 * np.log10(ratio)
    return db


def db2ratio(db):
    db = np.array(db, dtype='f')
    db = db.astype(float)
    ratio = 10 ** (db / 20.0)
    return ratio


class BandBassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')

    def run(self, signal):
        y = lfilter(self.b, self.a, signal)
        return y

######################################################
#
######################################################

vs = 340
fs = 300000
n = 7000
alpha = -1.3
ref_distance = 0.1


t_max = n / fs

bp_filter = BandBassFilter(30000,50000, fs)


data10 = np.load('1meter.npy')
data05 = np.load('0.5meter.npy')

data10 = bp_filter.run(data10)
data05 = bp_filter.run(data05)

time_axis = np.linspace(0,t_max, n)
distance_axis = time_axis * vs / 2
travel_distance_axis = distance_axis * 2

start_distance = 1.2
target_distance = 0.6


distance_shift = target_distance - start_distance
travel_distance_shift = 2 * distance_shift
new_distance_axis = distance_axis + distance_shift

pyplot.subplot(2,1,1)
pyplot.plot(distance_axis, data05)
pyplot.title('0.5 meter')
pyplot.subplot(2,1,2)
pyplot.plot(distance_axis, data10)
pyplot.title('1 meter')

# samples to shift
samples_shift = travel_distance_shift * fs / vs 
samples_shift = np.round(samples_shift)
samples_shift = int(samples_shift)

# atmospheric attenuation
attenuation_db = alpha * travel_distance_shift
attenuation_linear = db2ratio(attenuation_db)

# spreading attenuation

spreading = (1/distance_axis) * np.sqrt(1/distance_axis)#--> because 2x spreading
new_spreading = (1/new_distance_axis) * np.sqrt(1/new_distance_axis)
ratio = new_spreading / spreading



# ## Apply
# new_waveform = np.roll(data10, shift=samples_shift)
# new_waveform = new_waveform * attenuation_linear
# new_waveform = new_waveform * ratio

# pyplot.figure()
# pyplot.plot(travel_distance_axis, data05, alpha=0.5)
# pyplot.plot(travel_distance_axis, new_waveform, alpha=0.5)
# pyplot.legend(['0.5', 'new'])