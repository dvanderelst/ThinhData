import random

import numpy
from matplotlib import pyplot
from pyBat import Wiegrebe

from pyBat import Acoustics
from pyBat import Call

source = 'pd01'
freq_list = [40000]
azs = numpy.linspace(-30, 30, 10)

start_plant = 1000
end_plant = 3000

start_pole = 1000
end_pole = 2000

delta_pole = 300
jitter_plant = 300

samples = 7000

plants = numpy.load('data/planter-1m.npy')
plants = plants[:, start_plant:end_plant]
n_plants = plants.shape[0]

poles = numpy.load('data/pole-1m.npy')
poles = poles[:, start_pole:end_pole]
n_poles = poles.shape[0]

els = azs * 0
dists = (azs * 0) + 1
c = Call.Call(source=source, freq_list=freq_list, pitch=0)
x = c.call(azimuths=azs, elevations=els, distances=dists)

wiegrebe = Wiegrebe.ModelWiegrebe(300000, 40000, 3)

# %%
left_gains = x['directivity'][0]
right_gains = x['directivity'][1]
left_gains = Acoustics.db2ratio(left_gains)
right_gains = Acoustics.db2ratio(right_gains)

left_echo = numpy.zeros(samples)
right_echo = numpy.zeros(samples)
pole_echo = numpy.zeros(samples)

pole_echo[start_pole - delta_pole:end_pole - delta_pole] = poles[0, :]

for i in range(len(azs)):
    left_gain = left_gains[i]
    right_gain = right_gains[i]
    plant_index = random.choice(range(n_plants))
    signal = plants[plant_index, :]
    jitter = int(numpy.random.uniform(-jitter_plant, +jitter_plant))
    start = start_plant + jitter
    end = end_plant + jitter
    left_echo[start:end] = left_echo[start:end] + signal * left_gain
    right_echo[start:end] = right_echo[start:end] + signal * right_gain

left_and_pole = left_echo + pole_echo
right_and_pole = right_echo + pole_echo

pyplot.subplot(2, 2, 1)
pyplot.plot(left_echo)
pyplot.subplot(2, 2, 2)
pyplot.plot(right_echo)
pyplot.subplot(2, 2, 3)
pyplot.plot(pole_echo)
pyplot.show()

left_wiegrebe = wiegrebe.run_model(left_echo)
right_wiegrebe = wiegrebe.run_model(right_echo)

left_wiegrebe_total = wiegrebe.run_model(left_and_pole)
right_wiegrebe_total = wiegrebe.run_model(right_and_pole)

pyplot.subplot(1, 2, 1)
pyplot.plot(left_wiegrebe)
pyplot.plot(left_wiegrebe_total)
pyplot.subplot(1, 2, 2)
pyplot.plot(right_wiegrebe)
pyplot.plot(right_wiegrebe_total)
pyplot.show()





