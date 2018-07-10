import h5py

from fft import LowPassFilter
from mpi_signal_filter import SignalFilter

h5_path = 'giv_raw.h5'
h5_f = h5py.File(h5_path, mode='r+')
h5_grp = h5_f['Measurement_000/Channel_000']
h5_main = h5_grp['Raw_Data']

samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
num_spectral_pts = h5_main.shape[1]

frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
noise_tol = 1E-6

sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, verbose=True)
h5_filt_grp = sig_filt.compute()

# VERIFICATION here:
row_ind = 20
actual_line = h5_filt_grp['Filtered_Data'][row_ind]

h5_ref_path = '/home/syz/giv/pzt_nanocap_6_just_translation_filt_resh_copy.h5'
h5_ref_file = h5py.File(h5_ref_path, mode='r')
h5_ref_grp = h5_ref_file[h5_filt_grp.name]
ref_line = h5_ref_grp['Filtered_Data'][row_ind]

import numpy as np
print('Actual line close to reference:')
print(np.max(np.abs(actual_line - ref_line)))
print(np.allclose(actual_line, ref_line))

"""
single_AO = h5_grp['Spectroscopic_Values'][0, :500]

import numpy as np
row_ind = 20
# read data for a specific scan line
raw_line_resp = h5_main[row_ind]
# break this up into pixels:
raw_line_mat = np.reshape(raw_line_resp, (-1, single_AO.size))
filt_line_resp = h5_filt_grp['Filtered_Data'][row_ind]
filt_line_mat = np.reshape(filt_line_resp, (-1, single_AO.size))

import pyUSID as usid
fig, axes = usid.plot_utils.plot_curves(single_AO, [raw_line_mat, filt_line_mat], use_rainbow_plots=False, x_label='Bias (V)',
                                     y_label='Current (nA)', subtitle_prefix='Pixel', title=None, num_plots=9)
fig.savefig('result.png', format='png', )
savefig(os.path.join(other_figures_folder, file_name + '.png'), format='png', dpi=300)
"""

h5_f.close()
