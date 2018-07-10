import h5py
from mpi4py import MPI
from fft import LowPassFilter
from mpi_signal_filter import SignalFilter

h5_path = 'giv_raw.h5'
h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)

h5_grp = h5_f['Measurement_000/Channel_000']
h5_main = h5_grp['Raw_Data']

samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
num_spectral_pts = h5_main.shape[1]

frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
noise_tol = 1E-6

sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, verbose=False)
h5_filt_grp = sig_filt.compute()

# RuntimeError: Can't decrement id ref count (MPI_ERR_ARG: invalid argument of some other kind)
MPI.COMM_WORLD.barrier()
h5_f.close()
"""
try:
    h5_f.close()
    print('Rank {} managed to close'.format(MPI.COMM_WORLD.Get_rank()))
except RuntimeError:
    print('Rank {} experienced a run-time error when closing the file'.format(MPI.COMM_WORLD.Get_rank()))
"""
