import h5py
from mpi4py import MPI
from giv_bayesian_mpi import GIVBayesian

h5_path = 'giv_raw.h5'
h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)

h5_grp = h5_f['Measurement_000/Channel_000']

ex_freq = h5_grp.attrs['excitation_frequency_[Hz]']

h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

i_cleaner = GIVBayesian(h5_resh, ex_freq, 9, r_extra=110, num_x_steps=250, cores=3,
                           max_mem_mb=1024*10, verbose=True)
h5_bayes_grp = i_cleaner.compute()

# RuntimeError: Can't decrement id ref count (MPI_ERR_ARG: invalid argument of some other kind)
MPI.COMM_WORLD.barrier()
print('Rank {} passed the barrier'.format(MPI.COMM_WORLD.Get_rank()))
h5_f.close()
"""
try:
    h5_f.close()
    print('Rank {} managed to close'.format(MPI.COMM_WORLD.Get_rank()))
except RuntimeError:
    print('Rank {} experienced a run-time error when closing the file'.format(MPI.COMM_WORLD.Get_rank()))
"""
