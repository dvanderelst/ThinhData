import numpy as np
from scipy.signal import butter, lfilter


def closest(array, value):
    index = np.argmin(np.abs(array - value))
    return index


def ratio2db(ratio):
    db = 20 * np.log10(ratio)
    return db


def db2ratio(db):
    db = np.array(db, dtype='f')
    db = db.astype(float)
    ratio = 10 ** (db / 20.0)
    return ratio


def calculate_spreading(distances, outward=1, inward=0.5):
    # spherical spreading -> coeff =  1
    # cylindrical spreading --> coef = 0.5
    spreading_out = (1 / distances) ** outward
    spreading_in = (1 / distances) ** inward
    combined = spreading_in * spreading_out
    return combined


class BandBassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')

    def run(self, signal):
        y = lfilter(self.b, self.a, signal)
        return y