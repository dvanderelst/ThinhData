from EchoShifter import EchoShifter
import numpy as np
from matplotlib import pyplot
from pyBat import Wiegrebe

wiegrebe = Wiegrebe.ModelWiegrebe(sample_rate=300000, center=40000, bands=1)

for echo_nr in range(25):
    index_a = echo_nr
    index_b = echo_nr

    shifter = EchoShifter('data/planter-0.5m.npy', [0.25, 1], index=index_a, orginal_distance=0.5)
    result = shifter.shift(1, add_noise=False)
    mask = result['mask']

    reference_data = np.load('data/planter-1m.npy')[index_b, :]
    reference_data = shifter.bp_filter.run(reference_data) * mask

    shifted_wiegrebe = wiegrebe.run_model(result['new_wave'])
    reference_wiegrebe = wiegrebe.run_model(reference_data)

    output_name = 'output/' + str(index_a) + '_' + str(index_b) + '.png'

    pyplot.subplot(2,1,1)

    pyplot.plot(shifter.distance_axis, result['new_wave'], alpha=0.5)
    pyplot.plot(shifter.distance_axis, reference_data, alpha=0.5)

    pyplot.legend(['Shifted/Simulated', 'Measurement'])
    pyplot.xlim([0.2, 3])
    pyplot.ylim([-300, 300])
    pyplot.title('Wave forms')

    pyplot.subplot(2, 1, 2)

    pyplot.plot(shifter.distance_axis, shifted_wiegrebe)
    pyplot.plot(shifter.distance_axis, reference_wiegrebe)

    pyplot.legend(['Shifted/Simulated', 'Measurement'])

    pyplot.xlim([0.2, 3])
    pyplot.title('Wiegrebe model output')
    pyplot.tight_layout()
    pyplot.savefig(output_name)
    pyplot.close('all')



