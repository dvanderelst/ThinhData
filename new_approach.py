import numpy as np
from matplotlib import pyplot
from scipy.signal import butter, lfilter
import scipy

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

def calculate_spreading(distance):
    spreading_out = (1 / distance) ** 1
    spreading_in = (1 / distance) ** 0.5
    total_spreading = spreading_in * spreading_out
    return total_spreading


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
bp_filter = BandBassFilter(35000, 45000, fs)

move_distance = 0.6

extraction = [0.5, 1]

data = np.load('0.5meter.npy')
data = bp_filter.run(data)

time_axis = np.linspace(0,n / fs, n)
distance_axis = time_axis * vs / 2

pyplot.plot(distance_axis, data)
pyplot.show()

start_index = closest(distance_axis, extraction[0])
end_index = closest(distance_axis, extraction[1])
extracted_distances = distance_axis[start_index: end_index]
extracted_signal = data[start_index: end_index]

# atmospheric attenuation
attenuation_db = alpha * move_distance * 2
attenuation_linear = db2ratio(attenuation_db)

# spreading
new_extracted_distances = extracted_distances + move_distance

total_spreading = calculate_spreading(extracted_distances)
new_total_spreading = calculate_spreading(new_extracted_distances)
ratio =  new_total_spreading / total_spreading

# apply
new_signal = extracted_signal * attenuation_linear * ratio

#%%
comparison = np.load('1meter.npy')
comparison = bp_filter.run(comparison)

pyplot.subplot(2,1,1)
pyplot.plot(extracted_distances, extracted_signal)
pyplot.subplot(2,1,2)
pyplot.plot(new_extracted_distances, new_signal)
pyplot.show()

pyplot.plot(distance_axis, comparison)
pyplot.plot(new_extracted_distances, new_signal, alpha=0.25)
pyplot.show()

## get noise
comparison = np.load('1meter.npy')
comparison = bp_filter.run(comparison)
noise = comparison[6000:7000]
_, bins, _ = pyplot.hist(noise, bins=25, density=1)
mu, sigma = scipy.stats.norm.fit(noise)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
pyplot.plot(bins, best_fit_line)
pyplot.show()