import library
import numpy as np
from matplotlib import pyplot
import scipy

class EchoShifter:
    def __init__(self, filename, window):
        self.file_name = filename
        self.window = window
        self.original_distance = window[0]
        self.fs = 300000
        self.vs = 430
        self.alpha = -1.3
        self.samples_per_meter = self.fs / self.vs
        self.bp_filter = library.BandBassFilter(35000, 45000, self.fs)

        # Read and filter the recorded data
        self.raw = np.load(self.file_name)
        self.raw = self.bp_filter.run(self.raw)
        self.samples = self.raw.size
        self.noise_samples = self.raw[self.samples - 1000:]
        self.time_axis = np.linspace(0, self.samples / self.fs, self.samples)
        self.distance_axis = self.time_axis * self.vs / 2
        # Extract the relevant echo
        self.extracted_distances, self.extracted_signal = self.extract_window()
        # Estimate the noise distribution
        self.mu, self.sigma = scipy.stats.norm.fit(self.noise_samples)

    def plot_raw(self):
        pyplot.plot(self.distance_axis, self.raw)
        pyplot.title(self.file_name)
        pyplot.show()

    def plot_extracted(self):
        pyplot.plot(self.extracted_distances, self.extracted_signal)
        pyplot.title(self.file_name + ': ' + str(self.window))
        pyplot.show()

    def extract_window(self):
        start_distance = self.window[0]
        end_distance = self.window[1]
        distance_axis = self.distance_axis
        start_index = library.closest(distance_axis, start_distance)
        end_index = library.closest(distance_axis, end_distance)
        extracted_distances = distance_axis[start_index: end_index]
        extracted_signal = self.raw[start_index: end_index]
        return extracted_distances, extracted_signal

    def shift(self, new_distance, add_noise=True):
        one_way_shift = new_distance - self.original_distance
        two_way_shift = one_way_shift * 2

        # atmospheric attenuation
        attenuation_db = self.alpha * two_way_shift
        attenuation_linear = library.db2ratio(attenuation_db)

        # spreading
        new_extracted_distances = self.extracted_distances + one_way_shift

        total_spreading = library.calculate_spreading(self.extracted_distances)
        new_total_spreading = library.calculate_spreading(new_extracted_distances)
        spreading_linear = new_total_spreading / total_spreading

        # apply attenuations
        new_extracted_signal = self.extracted_signal * attenuation_linear * spreading_linear

        # generate shifted wave
        new_wave = np.zeros(self.samples)
        start_sample = 2 * self.samples_per_meter * new_distance
        start_sample = int(start_sample)
        end_sample = start_sample + len(new_extracted_signal)
        new_wave[start_sample:end_sample] = new_extracted_signal
        noise = np.zeros(self.samples)
        if add_noise: noise = np.random.normal(scale=self.sigma, size=self.samples)
        new_wave = new_wave + noise

        result = {}
        result['distances'] = new_extracted_distances
        result['signal'] = new_extracted_signal
        result['new_wave'] = new_wave
        result['distance_axis'] = self.distance_axis
        result['noise'] = noise
        return result


shifter = EchoShifter('0.5meter.npy', [0.7, 1.1])
shifter.plot_raw()
shifter.plot_extracted()
result = shifter.shift(1.4)

# test
data = np.load('1meter.npy')
data = shifter.bp_filter.run(data)
pyplot.plot(shifter.distance_axis, data)
pyplot.plot(result['distances'], result['signal'], alpha=0.5)
pyplot.show()

pyplot.plot(result['distance_axis'], result['new_wave'])
pyplot.show()
