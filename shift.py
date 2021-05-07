from EchoShifter import EchoShifter
import numpy as np
from matplotlib import pyplot

shifter = EchoShifter('data/planter-0.5m.npy', [0.25, 1], orginal_distance=0.5)
shifter.plot_extracted()
r = shifter.shift(1)
shifter.plot_result()

print(r['two_way_shift'])

# for echo_nr in range(25):
#     index_a = echo_nr
#     index_b = echo_nr
#
#     shifter = EchoShifter('planter.npy', [0.25, 1], index=index_a)
#     result = shifter.shift(0.75)
#
#     reference_data = np.load('data/planter-1m.npy')[index_b, :]
#     reference_data = shifter.bp_filter.run(reference_data)
#
#     output_name = 'output/' + str(index_a) + '_' + str(index_b) + '.png'
#     pyplot.plot(shifter.distance_axis, reference_data, alpha=0.5)
#     pyplot.plot(result['distance_axis'], result['new_wave'], alpha=0.5)
#     pyplot.xlim([0.2, 3])
#     pyplot.title(output_name)
#     pyplot.savefig(output_name)
#     pyplot.close('all')
#         #
#     # index_a = 0
# index_b = 0
#
# shifter = EchoShifter('planter.npy', [0.25, 1.5], echo_nr=index_a)
# shifter.plot_raw()
# shifter.plot_extracted()
# result = shifter.shift(0.75)
# pyplot.plot(result['distance_axis'], result['new_wave'], alpha=0.5)
# pyplot.plot(result['distances'], result['signal'], alpha=0.5)
# pyplot.show()
# # #
# #%% test
# data = np.load('planter-1m.npy')[index_b,:]
# data = shifter.bp_filter.run(data)
# pyplot.plot(shifter.distance_axis, data)
# pyplot.plot(result['distance_axis'], result['new_wave'], alpha=0.25)
# pyplot.xlim([0.25,2.5])
# pyplot.show()