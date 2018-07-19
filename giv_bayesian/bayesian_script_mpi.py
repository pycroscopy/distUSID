import multiprocessing

"""
It is crucial that NO OTHER package be imported till the process spawning method is set.
The next two lines must be left as is whether it is for single or multi-node computation.
There is a bug in the way numpy is linked with accelerator libraries in Unix based systems that can sometimes
cause multiprocessing / joblib to hang on numpy operations like dot. See these issues and links:
https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
https://github.com/joblib/joblib/issues/310
https://github.com/numpy/numpy/issues/5752

Solution:
http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux

So, keep everything under if __name__ == 'main'
"""
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import h5py
    from mpi4py import MPI
    from giv_bayesian_mpi import GIVBayesian

    h5_path = 'giv_raw.h5'
    h5_f = h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD)

    h5_grp = h5_f['Measurement_000/Channel_000']

    ex_freq = h5_grp.attrs['excitation_frequency_[Hz]']

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    i_cleaner = GIVBayesian(h5_resh, ex_freq, 9, r_extra=110, num_x_steps=250, verbose=True)
    h5_bayes_grp = i_cleaner.compute()

    h5_f.close()
