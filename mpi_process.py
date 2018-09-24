"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import os
import numpy as np
import psutil
import joblib
import time as tm
import h5py
import itertools
from numbers import Number

from multiprocessing import cpu_count

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        # mpi4py available but NOT called via mpirun or mpiexec => single node
        MPI = None
except ImportError:
    # mpi4py not even present! Single node by default:
    MPI = None

mpi_serial_warning = False

from pyUSID.io.hdf_utils import check_if_main, check_for_old, get_attributes
from pyUSID.io.usi_data import USIDataset
from pyUSID.io.io_utils import recommend_cpu_cores, get_available_memory, format_time, format_size

"""
For hyperthreaded applications: need to tack on the additional flag as shown below
No need to specify -n 4 or whatever if you want to use all available processors
$ mpirun -use-hwthread-cpus python hello_world.py 

Check the number of ranks per socket. If only 1 rank per socket - that rank is allowed to call joblib
Thus this paradigm will span the pure-mpi and mpi+joblib paradigm. Note that this does not prevent some sockets to run
in pure MPI mode while others run in MPI+joblib mode. Eventually, this should allow each rank to use jolib when the 
number of ranks in a given socket are noticeably less than the number of logical cores....

The naive approach will be to simply allow all ranks to write data directly to file
Forcing only a single rank within a socket may negate performance benefits
Writing out to separate files and then merging them later on is the most performant option

Look into sub-communication worlds that can create mini worlds instaed of the general COMM WOLRD
https://stackoverflow.com/questions/50900655/mpi4py-create-multiple-groups-and-scatter-from-each-group
https://www.bu.edu/pasi/files/2011/01/Lisandro-Dalcin-mpi4py.pdf
No work will be necessary to figure out the new ranking within the new communicator / group - automatically assigned 
from lowest value

set self.verbose = True for all master ranks. Won't need to worry about printing later on
Do we need a new variable called self._worker_ranks = [1,5,9, 13...] for master ranks? <-- this can save time and 
repeated book-keeping!

How much memory each rank can work with is a function of:
1. How much available memory this chip has
2. How many ranks are sharing this socket - Will need the new master per socket function's result

1. Find all unique ranks in the giant array of ranks
2. Create empty array to hold how much memory a rank can load
2. For each unique rank:
    a. Find available memory
    b. Find number of ranks that share this master
    c. Assign quotient to all ranks that share this socket

0. Do standard book-keeping of creating datasets etc.
1. Scatter position slices among all ranks
2. Instead of doing joblib either use a for-loop or the map-function
3. When it is time to write the results chunks back to file. 
    a. If not master -> send data to master
    b. If master -> gather from this smaller world and then write to file once. IF this is too much memory to handle, 
    then loop over each rank <-- how is this different from just looping over each rank within the new communicator and 
    asking it to write?:
        i. receive
        ii. write
        iii. repeat.
    A copy of the data will be made on Rank 0. ie - Rank 0 will have to hold N ranks worth of data. Meaning that each 
    rank can hold only around M/(2N) of data where M is the memory per node and N is the number of ranks per socket
    
http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
https://info.gwdg.de/~ceulig/docs-dev/doku.php?id=en:services:application_services:high_performance_computing:mpi4py
https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html

We know that just like the joblib mode, the HDF5 file is NOT corrupted if the job is preempted by the wall-time limit.
We need a more robust method for tracking what portions of the computation are completed.
The catch is that the previous method may have used serial / single node or a different number of ranks
Essentially, we would need to build a table of the pixels that still need to be computed and distribute this to the new 
ranks instead of a contiguous chunk from 0 to N pixels. This is drawing the Process class squarely into load management

Should one rank be spent on managing workers? We spend more time ahead of time to figure out which rank is responsible 
for what. Then all the ranks are on their own to do what they were told to.

Assuming that all ranks have the same amount of load, why ask them to start in evenly spaced portions of the dataset?
Given a dataset with 100 jobs and 4 ranks, currently, the ranks start as: 0, 25, 50, and 75

Now, tracking the completed portions, we have two challenges:
1. retaining what the previous jobs have completed
2. Having a contention-free way to track which rank has completed what
What should each rank do when it has finished one batch of computation?

HDF5 attributes CANNOT be modified (MPI or otherwise). They can be reassigned to a new value.
This means that  if multiple MPI ranks want to update an attribute, they will be overwriting each others changes
unless there is a lock and release / semaphore

just message passing does NOT prevent concurrency issues
Blocking will be wasting performance

Since concurrency on attributes cannot be controlled:
Option 1: create the same things as (indexed) datasets <--- can get ugly  very quickly

Option 2: Create a giant low precision dataset. Instead of storing indices, let each rank set the completed indices to 
True.  The problem is that the smallest precision is 1 byte and NOT 1 bit. Even boolean = 1 byte!
See - http://docs.h5py.org/en/latest/faq.html#faq
https://support.hdfgroup.org/HDF5/hdf5-quest.html#bool

https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/qFOGRTOxFTM
"""


def group_ranks_by_socket(verbose=False):
    """
    Groups MPI ranks in COMM_WORLD by socket. Another way to think about this is that it assigns a master rank for each
    rank such that there is a single master rank per socket (CPU). The results from this function can be used to split
    MPI communicators based on the socket for intra-node communication.

    This is necessary when wanting to carve up the memory for all ranks within a socket.
    This is also relevant when trying to bring down the number of ranks that are writing to the HDF5 file.
    This is all based on the premise that data analysis involves a fair amount of file writing and writing with
    3 ranks is a lot better than writing with 100 ranks. An assumption is made that the communication between the
    ranks within each socket would be faster than communicating across nodes / scokets. No assumption is made about the
    names of each socket

    Parameters
    ----------
    verbose : bool, optional
        Whether or not to print debugging statements

    Returns
    -------
    master_ranks : 1D unsigned integer numpy array
        Array with values that signify which rank a given rank should consider its master.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Step 1: Gather all the socket names:
    sendbuf = MPI.Get_processor_name()
    if verbose:
        print('Rank: ', rank, ', sendbuf: ', sendbuf)
    recvbuf = comm.allgather(sendbuf)
    if verbose and rank == 0:
        print('Rank: ', rank, ', recvbuf received: ', recvbuf)

    # Step 2: Find all unique socket names:
    recvbuf = np.array(recvbuf)
    unique_sockets = np.unique(recvbuf)
    if verbose and rank == 0:
        print('Unique sockets: {}'.format(unique_sockets))

    master_ranks = np.zeros(size, dtype=np.uint16)

    for item in unique_sockets:
        temp = np.where(recvbuf == item)[0]
        master_ranks[temp] = temp[0]

    if verbose and rank == 0:
        print('Parent rank for all ranks: {}'.format(master_ranks))

    return master_ranks


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples
    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
    Credits: @juanchopanza and @luca
    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list
    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


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

            if verbose:
                print("Rank {} of {} on {} sees {} logical cores on the socket".format(comm.Get_rank(), comm.Get_size(),
                                                                                       MPI.Get_processor_name(),
                                                                                       cpu_count()))

            # First, ensure that cores=logical cores in node. No point being economical / considerate
            cores = psutil.cpu_count()

            # It is sufficient if just one rank checks all this.
            if self.mpi_rank == 0:
                print('Working on {} ranks via MPI'.format(self.mpi_size))

            # Ensure that the file is opened in the correct comm or something
            if h5_main.file.driver != 'mpio':
                raise TypeError('The HDF5 file should have been opened with driver="mpio". Current driver = "{}"'
                                ''.format(h5_main.file.driver))

            """
            # Not sure how to check for this correctly
            messg = None
            try:
                if h5_main.file.comm != comm:
                    messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD. Currently comm={}'
                            ''.format(h5_main.file.comm)
            except AttributeError:
                messg = 'The HDF5 file should have been opened with comm=MPI.COMM_WORLD'
            if messg is not None:
                raise TypeError(messg)
            """

        else:
            print('No mpi4py found. Assuming single node computation')
            self.mpi_comm = None
            self.mpi_size = 1
            self.mpi_rank = 0

        # Checking if dataset is "Main"
        if not check_if_main(h5_main, verbose=verbose and self.mpi_rank == 0):
            raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')

        if MPI is not None:
            MPI.COMM_WORLD.barrier()
        # Not sure if we need a barrier here.

        # Saving these as properties of the object:
        self.h5_main = USIDataset(h5_main)
        self.verbose = verbose
        self._cores = None
        self._max_pos_per_read = None
        self._max_mem_mb = None

        # Now have to be careful here since the below properties are a function of the MPI rank
        self.__start_pos = None
        self.__rank_end_pos = None
        self.__end_pos = None
        self.__pixels_in_batch = None

        # Determining the max size of the data that can be put into memory
        # all ranks go through this and they need to have this value any
        self._set_memory_and_cores(cores=cores, mem=max_mem_mb)
        self.duplicate_h5_groups = []
        self.partial_h5_groups = []
        self.process_name = None  # Reset this in the extended classes
        self.parms_dict = None

        # The name of the HDF5 dataset that should be present to signify which positions have already been computed
        self.__status_dset_name = 'completed_positions'

        self._results = None
        self.h5_results_grp = None

        # Check to see if the resuming feature has been implemented:
        self.__resume_implemented = False
        try:
            self._get_existing_datasets()
        except NotImplementedError:
            if verbose and self.mpi_rank == 0:
                print('It appears that this class may not be able to resume computations')
        except:
            # NameError for variables that don't exist
            # AttributeError for self.var_name that don't exist
            # TypeError (NoneType) etc.
            self.__resume_implemented = True

        if self.mpi_rank == 0:
            print('Consider calling test() to check results before calling compute() which computes on the entire'
                  ' dataset and writes back to the HDF5 file')

        # DON'T check for duplicates since parms_dict has not yet been initialized.
        # Sub classes will check by themselves if they are interested.

    def __assign_job_indices(self):
        """
        Sets the start and end indices for each MPI rank
        """
        # First figure out what positions need to be computed
        self._compute_jobs = np.where(self._h5_status_dset[()] == 0)[0]
        if self.verbose and self.mpi_rank == 0:
            print('Among the {} positions in this dataset, the following positions need to be computed: {}'
                  '.'.format(self.h5_main.shape[0], self._compute_jobs))

        pos_per_rank = self._compute_jobs.size // self.mpi_size  # integer division
        if self.verbose and self.mpi_rank == 0:
            print('Each rank is required to work on {} of the {} (remaining) positions in this dataset'
                  '.'.format(pos_per_rank, self._compute_jobs.size))

        # The start and end indices now correspond to the indices in the incomplete jobs rather than the h5 dataset
        self.__start_pos = self.mpi_rank * pos_per_rank
        self.__rank_end_pos = (self.mpi_rank + 1) * pos_per_rank
        self.__end_pos = int(min(self.__rank_end_pos, self.__start_pos + self._max_pos_per_read))
        if self.mpi_rank == self.mpi_size - 1:
            # Force the last rank to go to the end of the dataset
            self.__rank_end_pos = self._compute_jobs.size

        if self.verbose:
            print('Rank {} will read positions {} to {} of {}'.format(self.mpi_rank, self.__start_pos,
                                                                      self.__rank_end_pos, self.h5_main.shape[0]))

    def _get_pixels_in_current_batch(self):
        """
        Returns the indices of the pixels that will be processed in this batch.

        Returns
        -------
        pixels_in_batch : numpy.ndarray
            1D array of unsigned integers denoting the pixels that will be read, processed, and written back to
        """
        return self.__pixels_in_batch

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
        duplicate_h5_groups : list of h5py.Group objects
            List of groups satisfying the above conditions
        """
        if self.verbose and self.mpi_rank == 0:
            print('Checking for duplicates:')

        # This list will contain completed runs only
        duplicate_h5_groups = check_for_old(self.h5_main, self.process_name, new_parms=self.parms_dict)
        partial_h5_groups = []

        # First figure out which ones are partially completed:
        if len(duplicate_h5_groups) > 0:
            for index, curr_group in enumerate(duplicate_h5_groups):
                """
                Earlier, we only checked the 'last_pixel' but to be rigorous we should check self.__status_dset_name
                The last_pixel attribute check may be deprecated in the future.
                Note that legacy computations did not have this dataset. We can add to partially computed datasets
                """
                if self.__status_dset_name in curr_group.keys():

                    # Case 1: Modern Process results:
                    status_dset = curr_group[self.__status_dset_name]

                    if not isinstance(status_dset, h5py.Dataset):
                        # We should not come here if things were implemented correctly
                        if self.mpi_rank == 0:
                            print('Results group: {} contained an object named: {} that should have been a dataset'
                                  '.'.format(curr_group, self.__status_dset_name))

                    if self.h5_main.shape[0] != status_dset.shape[0] or len(status_dset.shape) > 1 or \
                            status_dset.dtype != np.uint8:
                        if self.mpi_rank == 0:
                            print('Status dataset: {} was not of the expected shape or datatype'.format(status_dset))

                    # Finally, check how far the computation was completed.
                    if len(np.where(status_dset[()] == 0)[0]) == 0:
                        # remove from duplicates and move to partial
                        partial_h5_groups.append(duplicate_h5_groups.pop(index))
                        # Let's write the legacy attribute for safety
                        curr_group.attrs['last_pixel'] = self.h5_main.shape[0]
                        # No further checks necessary
                        continue

                # Case 2: Legacy results group:
                if 'last_pixel' not in curr_group.attrs.keys():
                    if self.mpi_rank == 0:
                        # Should not be coming here at all
                        print('Group: {} had neither the status HDF5 dataset or the legacy attribute: "last_pixel"'
                              '.'.format(curr_group))
                    # Not sure what to do with such groups. Don't consider them in the future
                    duplicate_h5_groups.pop(index)
                    continue

                # Finally, do the legacy test:
                if curr_group.attrs['last_pixel'] < self.h5_main.shape[0]:
                    # Should we create the dataset here, to make the group future-proof?
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
        h5_partial_group : h5py.Group object
            Group containing partially computed results
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

        self.h5_results_grp = h5_partial_group

    def _set_memory_and_cores(self, cores=None, mem=None):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods. This function can work with clusters with heterogeneous
        memory sizes (e.g. CADES SHPC Condo).

        Parameters
        ----------
        cores : uint, optional
            Default - 1
            How many cores to use for the computation
        mem : uint, optional
            Default - 1024
            The amount a memory in Mb to use in the computation
        """
        if MPI is None:
            min_free_cores = 1 + int(psutil.cpu_count() > 4)

            if cores is None:
                self._cores = max(1, psutil.cpu_count() - min_free_cores)
            else:
                if not isinstance(cores, int):
                    raise TypeError('cores should be an integer but got: {}'.format(cores))
                cores = int(abs(cores))
                self._cores = max(1, min(psutil.cpu_count(), cores))
            socket_master = 0
            ranks_per_socket = 1
        else:
            # user-provided input cores will simply be ignored in an effort to use the entire CPU
            ranks_by_socket = group_ranks_by_socket(verbose=self.verbose)
            socket_master = ranks_by_socket[self.mpi_rank]
            # which ranks in this socket?
            ranks_on_this_socket = np.where(ranks_by_socket == socket_master)[0]
            # how many in this socket?
            ranks_per_socket = ranks_on_this_socket.size
            # Force usage of all available memory
            mem = None
            self._cores = self.__cores_per_rank = psutil.cpu_count() // ranks_per_socket

        # TODO: Convert all to bytes!
        _max_mem_mb = get_available_memory() / 1024 ** 2  # in MB
        if mem is None:
            mem = _max_mem_mb
        else:
            if not isinstance(mem, int):
                raise TypeError('mem must be a whole number')
            mem = abs(mem)

        self._max_mem_mb = min(_max_mem_mb, mem)

        # Remember that multiple processes (either via MPI or joblib) will share this socket
        max_data_chunk = self._max_mem_mb / (self._cores * ranks_per_socket)

        # Now calculate the number of positions OF RAW DATA ONLY that can be stored in memory in one go PER RANK
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1024 ** 2
        self._max_pos_per_read = int(np.floor(max_data_chunk / mb_per_position))

        if self.verbose and self.mpi_rank == socket_master:
            # expected to be the same for all ranks so just use this.
            print('Rank {} - on socket with {} logical cores and {} avail. RAM shared by {} ranks each given {} cores'
                  '.'.format(socket_master, psutil.cpu_count(), format_size(_max_mem_mb * 1024**2, 2), ranks_per_socket,
                             self._cores))
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

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
        if self.__start_pos < self.__rank_end_pos:
            self.__end_pos = int(min(self.__rank_end_pos, self.__start_pos + self._max_pos_per_read))

            # DON'T DIRECTLY apply the start and end indices anymore to the h5 dataset. Find out what it means first
            self.__pixels_in_batch = self._compute_jobs[self.__start_pos: self.__end_pos]
            self.data = self.h5_main[self.__pixels_in_batch, :]
            if self.verbose:
                print('Rank {} - Read positions: {}'.format(self.mpi_rank, self.__pixels_in_batch, self.__rank_end_pos))

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
        self.__start_pos = self.__end_pos
        # This line can remain as is
        raise NotImplementedError('Please override the _set_results specific to your process')

    def _create_results_datasets(self):
        """
        Process specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        raise NotImplementedError('Please override the _create_results_datasets specific to your process')

    def __create_compute_status_dataset(self):
        """
        Creates a dataset that keeps track of what pixels / rows have already been computed. Users are not expected to
        extend / modify this function.
        """
        # Check to make sure that such a group doesn't already exist
        if self.__status_dset_name in self.h5_results_grp.keys():
            self._h5_status_dset = self.h5_results_grp[self.__status_dset_name]
            if not isinstance(self._h5_status_dset, h5py.Dataset):
                raise ValueError('Provided results group: {} contains an expected object ({}) that is not a dataset'
                                 '.'.format(self.h5_results_grp, self._h5_status_dset))
            if self.h5_main.shape[0] != self._h5_status_dset.shape[0] or len(self._h5_status_dset.shape) > 1 or \
                    self._h5_status_dset.dtype != np.uint8:
                if self.mpi_rank == 0:
                    raise ValueError('Status dataset: {} was not of the expected shape or datatype'
                                     '.'.format(self._h5_status_dset))
        else:
            self._h5_status_dset = self.h5_results_grp.create_dataset(self.__status_dset_name, dtype=np.uint8,
                                                                      shape=(self.h5_main.shape[0],))
            #  Could be fresh computation or resuming from a legacy computation
            if 'last_pixel' in self.h5_results_grp.attrs.keys():
                completed_pixels = self.h5_results_grp.attrs['last_pixel']
                if completed_pixels > 0:
                    self._h5_status_dset[:completed_pixels] = 1

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
        # cores = number of processes / rank here
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
        h5_results_grp : h5py.Group object
            Group containing all the results
        """

        class SimpleFIFO(object):
            """
            Simple class that maintains a moving average of some numbers.
            """

            def __init__(self, length=5):
                """
                Create a SimpleFIFO object

                Parameters
                ----------
                length : unsigned integer
                    Number of values that need to be maintained for the moving average
                """
                self.__queue = list()
                if not isinstance(length, int):
                    raise TypeError('length must be a positive integer')
                if length <= 0:
                    raise ValueError('length must be a positive integer')
                self.__max_length = length
                self.__count = 0

            def put(self, item):
                """
                Adds the item to the internal queue. If the size of the queue exceeds its capacity, the oldest
                item is removed.

                Parameters
                ----------
                item : float or int
                    Any real valued number
                """
                if (not isinstance(item, Number)) or isinstance(item, complex):
                    raise TypeError('Provided item: {} is not a Number'.format(item))
                self.__queue.append(item)
                self.__count += 1
                if len(self.__queue) > self.__max_length:
                    _ = self.__queue.pop(0)

            def get_mean(self):
                """
                Returns the average of the elements within the queue

                Returns
                -------
                avg : number.Number
                    Mean of all elements within the queue
                """
                return np.mean(self.__queue)

            def get_cycles(self):
                """
                Returns the number of items that have been added to the queue in total

                Returns
                -------
                count : int
                    number of items that have been added to the queue in total
                """
                return self.__count

        if not override:
            if len(self.duplicate_h5_groups) > 0:
                if self.mpi_rank == 0:
                    print('Returned previously computed results at ' + self.duplicate_h5_groups[-1].name)
                return self.duplicate_h5_groups[-1]
            elif len(self.partial_h5_groups) > 0:
                if self.mpi_rank == 0:
                    print('Resuming computation in group: ' + self.partial_h5_groups[-1].name)
                self.use_partial_computation()

        if self.h5_results_grp is None:
            # starting fresh
            if self.verbose and self.mpi_rank == 0:
                print('Creating HDF5 group and datasets to hold results')
            self._create_results_datasets()
        else:
            # resuming from previous checkpoint
            if self.verbose and self.mpi_rank == 0:
                print('Resuming computation')
            self._get_existing_datasets()

        self.__create_compute_status_dataset()
        self.__assign_job_indices()

        # Not sure if this is necessary but I don't think it would hurt either
        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

        compute_times = SimpleFIFO(5)
        write_times = SimpleFIFO(5)
        orig_rank_start = self.__start_pos

        if self.mpi_rank == 0 and self.mpi_size == 1:
            if self.__resume_implemented:
                print('\tThis class (likely) supports interruption and resuming of computations!\n'
                      '\tIf you are operating in a python console, press Ctrl+C or Cmd+C to abort\n'
                      '\tIf you are in a Jupyter notebook, click on "Kernel">>"Interrupt"\n'
                      '\tIf you are operating on a cluster and your job gets killed, re-run the job to resume\n')
            else:
                print('\tThis class does NOT support interruption and resuming of computations.\n'
                      '\tIn order to enable this feature, simply implement the _get_existing_datasets() function')

        self._read_data_chunk()
        
        while self.data is not None:

            t_start_1 = tm.time()

            self._unit_computation(*args, **kwargs)

            comp_time = np.round(tm.time() - t_start_1, decimals=2)  # in seconds
            time_per_pix = comp_time / (self.__end_pos - self.__start_pos)
            compute_times.put(time_per_pix)

            if self.verbose:
                print('Rank {} - computed chunk in {} or {} per pixel. Average: {} per pixel'
                      '.'.format(self.mpi_rank, format_time(comp_time), format_time(time_per_pix),
                                 format_time(compute_times.get_mean())))

            t_start_2 = tm.time()
            self._write_results_chunk()
            # NOW, update the positions. Users are NOT allowed to touch start and end pos
            self.__start_pos = self.__end_pos
            # Leaving in this provision that will allow restarting of processes
            self.h5_results_grp.attrs['last_pixel'] = self.__end_pos
            self.h5_main.file.flush()

            dump_time = np.round(tm.time() - t_start_2, decimals=2)
            write_times.put(dump_time / (self.__end_pos - self.__start_pos))

            if self.verbose:
                print('Rank {} - wrote its {} pixel chunk in {}'.format(self.mpi_rank,
                                                                        self.__end_pos - self.__start_pos,
                                                                        format_time(dump_time)))

            time_remaining = (self.__rank_end_pos - self.__end_pos) * \
                             (compute_times.get_mean() + write_times.get_mean())

            if self.verbose or self.mpi_rank == 0:
                percent_complete = int(100 * (self.__end_pos - orig_rank_start) /
                                       (self.__rank_end_pos - orig_rank_start))
                print('Rank {} - {}% complete. Time remaining: {}'.format(self.mpi_rank, percent_complete,
                                                                          format_time(time_remaining)))

            # All ranks should mark the pixels for this batch as completed. 'last_pixel' attribute will be updated later
            # Setting each section to 1 independently
            for section in to_ranges(self.__pixels_in_batch):
                self._h5_status_dset[section[0]: section[1]+1] = 1

            self._read_data_chunk()

        if self.verbose:
            print('Rank {} - Finished computing all jobs!'.format(self.mpi_rank))

        self.mpi_comm.barrier()
        if self.mpi_rank == 0:
            print('Finished processing the entire dataset!')

        # Update the 'last_pixel' attribute here:
        if self.mpi_rank == 0:
            self.h5_results_grp.attrs['last_pixel'] = self.h5_main.shape[0]

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
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    cores = recommend_cpu_cores(data.shape[0],
                                requested_cores=cores,
                                lengthy_computation=lengthy_computation,
                                verbose=verbose)

    """
    Disable threading since we tend to use MPI / multiprocessing / joblib.
    Not doing so has resulted in dramatically poorer performance due to competition between threads and processes
    Question is whether or not these variables should be reset to their prior values after computation.
    """
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[var] = '1'

    if verbose:
        print('Rank {} starting computing on {} cores (requested {} cores)'.format(rank, cores, req_cores))

    if cores > 1:
        values = [joblib.delayed(func)(x, *func_args, **func_kwargs) for x in data]
        results = joblib.Parallel(n_jobs=cores)(values)

        # Finished reading the entire data set
        print('Rank {} finished parallel computation'.format(rank))

    else:
        if verbose:
            print("Rank {} computing serially ...".format(rank))
        results = [func(vector, *func_args, **func_kwargs) for vector in data]

    return results
