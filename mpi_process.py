"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import psutil
import joblib
import time as tm

from multiprocessing import cpu_count
import socket
try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        # mpi4py available but NOT called via mpirun or mpiexec => single node
        MPI = None
except ImportError:
    # mpi4py not even present! Single node by default:
    MPI = None

from pyUSID.io.hdf_utils import check_if_main, check_for_old, get_attributes
from pyUSID.io.usi_data import USIDataset
from pyUSID.io.io_utils import recommend_cpu_cores, get_available_memory, format_time


class Process(object):
    """
    Encapsulates the typical steps performed when applying a processing function to  a dataset.
    """

    def __init__(self, h5_main, cores=None, max_mem_mb=4*1024, verbose=False):
        """
        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        cores : uint, optional
            Default - all available cores - 2
            How many cores to use for the computation
        max_mem_mb : uint, optional
            How much memory to use for the computation.  Default 1024 Mb
        verbose : Boolean, (Optional, default = False)
            Whether or not to print debugging statements
        """

        if h5_main.file.mode != 'r+':
            raise TypeError('Need to ensure that the file is in r+ mode to write results back to the file')

        if MPI is not None:
            # If we came here then, the user has intentionally asked for multi-node computation
            comm = MPI.COMM_WORLD
            self.mpi_comm = comm
            self.mpi_rank = comm.Get_rank()
            self.mpi_size = comm.Get_size()

            print("Rank {} of {} on {} sees {} logical cores on the socket".format(comm.Get_rank(), comm.Get_size(), socket.gethostname(), cpu_count()))

            # First, ensure that cores=logical cores in node. No point being economical / considerate
            cores = psutil.cpu_count()

            # It is sufficient if just one rank checks all this.
            if verbose and self.mpi_rank == 0:
                print('Working on {} nodes via MPI'.format(self.mpi_size))

            # Ensure that the file is opened in the correct comm or something
            if h5_main.file.driver != 'mpio':
                raise TypeError('The HDF5 file should have been opened with driver="mpio". Current driver = "{}"'.format(h5_main.file.driver))

            """
            # Not sure how to check for this correctly
            messg = None
            try:
                if h5_main.file.comm != comm:
                    messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD. Currently comm={}'.format(h5_main.file.comm)
            except AttributeError:
                messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD'
            if messg is not None:
                raise TypeError(messg)
            """

        else:
            if verbose:
                print('No mpi4py found. Asssuming single node computation')
            self.mpi_comm = None
            self.mpi_size = 1
            self.mpi_rank = 0

        # Checking if dataset is "Main"
        if self.mpi_rank == 0:
            if not check_if_main(h5_main, verbose=verbose):
                raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')
        # Not sure if we need a barrier here.

        # Saving these as properties of the object:
        self.h5_main = USIDataset(h5_main)
        self.verbose = verbose
        self._max_pos_per_read = None
        self._max_mem_mb = None

        # Now have to be careful here since the below properties are a function of the MPI rank
        self._start_pos = None
        self._rank_end_pos = None
        self._end_pos = None
        self.__assign_job_indices(start=0)

        # Determining the max size of the data that can be put into memory
        # all ranks go through this and they need to have this value any
        self._set_memory_and_cores(cores=cores, mem=max_mem_mb)
        self.duplicate_h5_groups = []
        self.partial_h5_groups = []
        self.process_name = None  # Reset this in the extended classes
        self.parms_dict = None

        self._results = None
        self.h5_results_grp = None

        if self.mpi_rank == 0:
            print('Consider calling test() to check results before calling compute() which computes on the entire'
                  ' dataset and writes back to the HDF5 file')

        # DON'T check for duplicates since parms_dict has not yet been initialized.
        # Sub classes will check by themselves if they are interested.

    def __assign_job_indices(self, start=0):
        """
        Sets the start and end indices for each MPI rank

        Parameters
        ----------
        start : uint (optional), default = 0
            Position index from which to start computing. Default assumes a fresh computation (start = 0)
        """
        pos_per_rank = (self.h5_main.shape[0] - start) // self.mpi_size  # integer division
        if self.verbose and self.mpi_rank==0:
            print('Each rank is required to work on {} of the {} positions in this dataset'.format(pos_per_rank, self.h5_main.shape[0] - start))
        self._start_pos = start + self.mpi_rank * pos_per_rank
        self._rank_end_pos = start + (self.mpi_rank+1) * pos_per_rank
        self._end_pos = self._rank_end_pos
        if self.mpi_rank == self.mpi_size - 1:
            # Force the last rank to go to the end of the dataset
            self._rank_end_pos = self.h5_main.shape[0]
        if self.verbose:
            print('Rank {} will read positions {} to {} of {}'.format(self.mpi_rank, self._start_pos, self._rank_end_pos, self.h5_main.shape[0]))

    def test(self, **kwargs):
        """
        Tests the process on a subset (for example a pixel) of the whole data. The class can be reinstantiated with
        improved parameters and tested repeatedly until the user is content, at which point the user can call
        compute() on the whole dataset. This is not a function that is expected to be called in mpi
        Parameters
        ----------
        kwargs - dict, optional
            keyword arguments to test the process
        Returns
        -------
        """
        # All children classes should call super() OR ensure that they only work for self.mpi_rank == 0
        raise NotImplementedError('test_on_subset has not yet been implemented')

    def _check_for_duplicates(self):
        """
        Checks for instances where the process was applied to the same dataset with the same parameters
        Returns
        -------
        duplicate_h5_groups : list of h5py.Datagroup objects
            List of groups satisfying the above conditions
        """
        if self.verbose and self.mpi_rank == 0:
            print('Checking for duplicates:')

        duplicate_h5_groups = check_for_old(self.h5_main, self.process_name, new_parms=self.parms_dict)
        partial_h5_groups = []

        # First figure out which ones are partially completed:
        if len(duplicate_h5_groups) > 0:
            for index, curr_group in enumerate(duplicate_h5_groups):
                if curr_group.attrs['last_pixel'] < self.h5_main.shape[0]:
                    # remove from duplicates and move to partial
                    partial_h5_groups.append(duplicate_h5_groups.pop(index))

        if len(duplicate_h5_groups) > 0 and self.mpi_rank == 0:
            print('Note: ' + self.process_name + ' has already been performed with the same parameters before. '
                                                 'These results will be returned by compute() by default. '
                                                 'Set override to True to force fresh computation')
            print(duplicate_h5_groups)

        if partial_h5_groups and self.mpi_rank == 0:
            print('Note: ' + self.process_name + ' has already been performed PARTIALLY with the same parameters. '
                                                 'compute() will resuming computation in the last group below. '
                                                 'To choose a different group call use_patial_computation()'
                                                 'Set override to True to force fresh computation or resume from a '
                                                 'data group besides the last in the list.')
            print(partial_h5_groups)

        return duplicate_h5_groups, partial_h5_groups

    def use_partial_computation(self, h5_partial_group=None):
        """
        Extracts the necessary parameters from the provided h5 group to resume computation
        Parameters
        ----------
        h5_partial_group : h5py.Datagroup object
            Datagroup containing partially computed results
        """
        # Attempt to automatically take partial results
        if h5_partial_group is None:
            if len(self.partial_h5_groups) < 1:
                raise ValueError('No group was found with partial results and no such group was provided')
            h5_partial_group = self.partial_h5_groups[-1]
        else:
            # Make sure that this group is among the legal ones already discovered:
            if h5_partial_group not in self.partial_h5_groups:
                raise ValueError('Provided group does not appear to be in the list of discovered groups')

        self.parms_dict = get_attributes(h5_partial_group)

        # Be careful in assigning the start and end positions - these will be per rank!
        self.__assign_job_indices(start=self.parms_dict.pop('last_pixel'))

        self.h5_results_grp = h5_partial_group

    def _set_memory_and_cores(self, cores=None, mem=None):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.
        Parameters
        ----------
        cores : uint, optional
            Default - 1
            How many cores to use for the computation
        mem : uint, optional
            Default - 1024
            The amount a memory in Mb to use in the computation
        """
        min_free_cores = 1 + int(psutil.cpu_count() > 4)

        if cores is None:
            self._cores = max(1, psutil.cpu_count() - min_free_cores)
        else:
            if not isinstance(cores, int):
                raise TypeError('cores should be an integer but got: {}'.format(cores))
            cores = int(abs(cores))
            self._cores = max(1, min(psutil.cpu_count(), cores))

        _max_mem_mb = get_available_memory() / 1E6  # in MB
        if mem is None:
            mem = _max_mem_mb
        else:
            if not isinstance(mem, int):
                raise TypeError('mem must be a whole number')
            mem = abs(mem)

        self._max_mem_mb = min(_max_mem_mb, mem)

        max_data_chunk = self._max_mem_mb / self._cores

        # Now calculate the number of positions that can be stored in memory in one go.
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1e6
        self._max_pos_per_read = int(np.floor(max_data_chunk / mb_per_position))

        if self.verbose and self.mpi_rank == 0:
            # expected to be the same for all ranks so just use this.
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))
            print('Allowed to use up to', str(self._cores), 'cores and', str(self._max_mem_mb), 'MB of memory')

    @staticmethod
    def _map_function(*args, **kwargs):
        """
        The function that manipulates the data on a single instance (position). This will be used by _unit_computation()
        to process a chunk of data in parallel
        Parameters
        ----------
        args : list
            arguments to the function in the correct order
        kwargs : dictionary
            keyword arguments to the function
        Returns
        -------
        object
        """
        raise NotImplementedError('Please override the _unit_function specific to your process')

    def _read_data_chunk(self):
        """
        Reads a chunk of data for the intended computation into memory
        """
        if self._start_pos < self._rank_end_pos:
            self._end_pos = int(min(self._rank_end_pos, self._start_pos + self._max_pos_per_read))
            self.data = self.h5_main[self._start_pos:self._end_pos, :]
            if self.verbose:
                print('Rank {} - Read positions {} to {}. Need to read till {}'.format(self.mpi_rank, self._start_pos, self._end_pos, self._rank_end_pos))

            # DON'T update the start position

        else:
            if self.verbose:
                print('Rank {} - Finished reading all data!'.format(self.mpi_rank))
            self.data = None

    def _write_results_chunk(self):
        """
        Writes the computed results into appropriate datasets.
        This needs to be rewritten since the processed data is expected to be at least as large as the dataset
        """
        # Now update the start position
        self._start_pos = self._end_pos
        raise NotImplementedError('Please override the _set_results specific to your process')

    def _create_results_datasets(self):
        """
        Process specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        raise NotImplementedError('Please override the _create_results_datasets specific to your process')

    def _get_existing_datasets(self):
        """
        The purpose of this function is to allow processes to resume from partly computed results
        Start with self.h5_results_grp
        """
        raise NotImplementedError('Please override the _get_existing_datasets specific to your process')

    def _unit_computation(self, *args, **kwargs):
        """
        The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
        as well as multiple calls to parallel_compute if necessary
        """
        # TODO: Try to use the functools.partials to preconfigure the map function
        self._results = parallel_compute(self.data, self._map_function, cores=self._cores,
                                         lengthy_computation=False,
                                         func_args=args, func_kwargs=kwargs,
                                         verbose=self.verbose)

    def compute(self, override=False, *args, **kwargs):
        """
        Creates placeholders for the results, applies the unit computation to chunks of the dataset
        Parameters
        ----------
        override : bool, optional. default = False
            By default, compute will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.
        args : list
            arguments to the mapped function in the correct order
        kwargs : dictionary
            keyword arguments to the mapped function
        Returns
        -------
        h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results
        """
        if not override:
            if len(self.duplicate_h5_groups) > 0:
                print('Returned previously computed results at ' + self.duplicate_h5_groups[-1].name)
                return self.duplicate_h5_groups[-1]
            elif len(self.partial_h5_groups) > 0:
                print('Resuming computation in group: ' + self.partial_h5_groups[-1].name)
                self.use_partial_computation()

        if self.h5_results_grp is None:
            # starting fresh
            if self.verbose:
                print('Creating datagroup and datasets')
            self._create_results_datasets()
        else:
            # resuming from previous checkpoint
            if self.verbose:
                print('Resuming computation')
            self._get_existing_datasets()

        # Not sure if this is necessary but I don't think it would hurt either
        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

        time_per_pix = 0
        num_pos = self._rank_end_pos - self._start_pos
        orig_start_pos = self._start_pos

        # TODO: Need to find a nice way of figuring out if a process has implemented the partial feature.
        if self.mpi_rank == 0:
            print('You maybe able to abort this computation at any time and resume at a later time!\n'
                  '\tIf you are operating in a python console, press Ctrl+C or Cmd+C to abort\n'
                  '\tIf you are in a Jupyter notebook, click on "Kernel">>"Interrupt"')

        self._read_data_chunk()
        while self.data is not None:

            t_start = tm.time()

            self._unit_computation(*args, **kwargs)

            tot_time = np.round(tm.time() - t_start, decimals=2)
            if self.verbose:
                print('Rank {} - parallel computed chunk in {} or {} per pixel'.format(self.mpi_rank, format_time(tot_time),
                                                                             format_time(
                                                                                 tot_time / self.data.shape[0])))
            if self._start_pos == orig_start_pos:
                time_per_pix = tot_time / (self._rank_end_pos - orig_start_pos)  # in seconds
            else:
                time_remaining = (num_pos - (self._rank_end_pos - self._start_pos)) * time_per_pix  # in seconds
                print('Rank {} - Time remaining: {}'.format(self.mpi_rank, format_time(time_remaining)))

            self._write_results_chunk()
            self._read_data_chunk()

        print('Rank {} - Finished computing all jobs!'.format(self.mpi_rank))

        return self.h5_results_grp


def parallel_compute(data, func, cores=1, lengthy_computation=False, func_args=None, func_kwargs=None, verbose=False):
    """
    Computes the provided function using multiple cores using the joblib library

    Parameters
    ----------
    data : numpy.ndarray
        Data to map function to. Function will be mapped to the first axis of data
    func : callable
        Function to map to data
    cores : uint, optional
        Number of logical cores to use to compute
        Default - 1 (serial computation)
    lengthy_computation : bool, optional
        Whether or not each computation is expected to take substantial time.
        Sometimes the time for adding more cores can outweigh the time per core
        Default - False
    func_args : list, optional
        arguments to be passed to the function
    func_kwargs : dict, optional
        keyword arguments to be passed onto function
    verbose : bool, optional. default = False
        Whether or not to print statements that aid in debugging
    Returns
    -------
    results : list
        List of computational results
    """

    if not callable(func):
        raise TypeError('Function argument is not callable')
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')
    if func_args is None:
        func_args = list()
    else:
        if isinstance(func_args, tuple):
            func_args = list(func_args)
        if not isinstance(func_args, list):
            raise TypeError('Arguments to the mapped function should be specified as a list')
    if func_kwargs is None:
        func_kwargs = dict()
    else:
        if not isinstance(func_kwargs, dict):
            raise TypeError('Keyword arguments to the mapped function should be specified via a dictionary')
    req_cores = cores
    cores = recommend_cpu_cores(data.shape[0],
                                requested_cores=cores,
                                lengthy_computation=lengthy_computation,
                                verbose=verbose)

    print('Starting computing on {} cores (requested {} cores)'.format(cores, req_cores))

    if cores > 1:
        values = [joblib.delayed(func)(x, *func_args, **func_kwargs) for x in data]
        results = joblib.Parallel(n_jobs=cores)(values)

        # Finished reading the entire data set
        print('Finished parallel computation')

    else:
        print("Computing serially ...")
        results = [func(vector, *func_args, **func_kwargs) for vector in data]

    return results
