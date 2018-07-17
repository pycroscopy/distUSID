"""
Created on Oct 14, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import sys
import os
import getopt
import h5py
import numpy as np
from mpi4py import MPI
# import matplotlib.pyplot as plt

def gen_batches(n, batch_size, start=0):
    """
    Copied from scikit-learn

    Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.

    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    """
    stop = start+n
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end

    if start < stop:
        yield slice(start, stop)


def gen_batches_mpi(n_items, n_batches):
    """
    Modified version of gen_batches

    Generator to create `n_batches` slices, from 0 to n_items.

    Some slices may contain less more elements than others, when batch_size
    does not divide evenly into n_items.

    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    """

    batch_size = 0
    rem = n_items
    while rem > n_batches:
        batch_size += n_items/n_batches
        rem = n_items % n_batches

    start = 0
    for _ in range(n_batches):
        end = start + batch_size
        if rem > 0:
            end += 1
            rem -= 1
        yield(slice(start, end))

        start = end


def calc_chunks(dimensions, data_size, unit_chunks=None, max_chunk_mem=10240):
    """
    Calculate the chunk size for the HDF5 dataset based on the dimensions and the
    maximum chunk size in memory

    Parameters
    ----------
    dimensions : array_like of int
        Shape of the data to be chunked
    data_size : int
        Size of an entry in the data in bytes
    unit_chunks : array_like of int, optional
        Unit size of the chunking in each dimension.  Must be the same size as
        the shape of `ds_main`.  Default None, `unit_chunks` is set to 1 in all
        dimensions
    max_chunk_mem : int, optional
        Maximum size of the chunk in memory in bytes.  Default 10240b or 10kb

    Returns
    -------
    chunking : tuple of int
        Calculated maximum size of a chunk in each dimension that is as close to the
        requested `max_chunk_mem` as posible while having steps based on the input
        `unit_chunks`.
    """
    '''
    Ensure that dimensions is an array
    '''
    dimensions = np.asarray(dimensions, dtype=np.uint)
    '''
    Set the unit_chunks to all ones if not given.  Ensure it is an array if it is.
    '''
    if unit_chunks is None:
        unit_chunks = np.ones_like(dimensions)
    else:
        unit_chunks = np.asarray(unit_chunks, dtype=np.uint)

    if unit_chunks.shape != dimensions.shape:
        raise ValueError('Unit chunk size must have the same shape as the input dataset.')

    '''
    Save the original size of unit_chunks to use for incrementing the chunk size during
     loop
    '''
    base_chunks = unit_chunks

    '''
    Loop until chunk_size is greater than the maximum chunk_mem or the chunk_size is equal to
    that of dimensions
    '''
    while np.prod(unit_chunks) * data_size <= max_chunk_mem:
        '''
        Check if all chunk dimensions are greater or equal to the
        actual dimensions.  Exit the loop if true.
        '''
        if np.all(unit_chunks >= dimensions):
            break

        '''
        Find the index of the next chunk to be increased and increment it by the base_chunk
        size
        '''
        ichunk = np.argmax(dimensions / unit_chunks)
        unit_chunks[ichunk] += base_chunks[ichunk]

    '''
    Ensure that the size of the chunks is between one and the dimension size.
    '''
    unit_chunks = np.clip(unit_chunks, np.ones_like(unit_chunks), dimensions)

    chunking = tuple(unit_chunks)

    return chunking


def clean_and_build_components(h5_svd, comm, comp_slice=None):
    """
    Rebuild the Image from the PCA results on the windows
    Optionally, only use components less than n_comp.

    Parameters
    ----------
    h5_svd : hdf5 Group, optional
        Group containing the results from SVD on windowed data
    components: {int, iterable of int, slice} optional
        Defines which components to keep

        Input Types
        integer : Components less than the input will be kept
        length 2 iterable of integers : Integers define start and stop of component slice to retain
        other iterable of integers or slice : Selection of component indices to retain

    Returns
    -------
    h5_clean : hdf5 Dataset
        dataset with the cleaned image
    """

    mpi_size = comm.size
    mpi_rank = comm.rank

    try:
        h5_S = h5_svd['S']
        h5_U = h5_svd['U']
        h5_V = h5_svd['V']

        print('Cleaning the image by removing unwanted components.')
    except:
        raise

    comp_slice = get_component_slice(comp_slice)
    '''
    Get basic windowing information from attributes of
    h5_win_group
    '''
    h5_win_group = h5_svd.parent
    h5_pos = h5_win_group['Position_Indices']

    im_x = h5_win_group.attrs['image_x']
    im_y = h5_win_group.attrs['image_y']
    win_x = h5_win_group.attrs['win_x']
    win_y = h5_win_group.attrs['win_y']

    win_name = h5_win_group.name.split('/')[-1]
    basename = win_name.split('-')[0]
    h5_source_image = h5_win_group.parent[basename]

    '''
    Create slice object from the positions
    '''
    # Get batches of positions for each process
    n_pos = h5_pos.shape[0]
    win_batches = gen_batches_mpi(n_pos, mpi_size)

    # Create array of windows for each process
    if mpi_rank == 0:
        print '# of positions:  {}'.format(n_pos)

        windows = [h5_pos[my_batch] for my_batch in win_batches]

        for rank in range(mpi_size):
            print 'Process {} has {} windows.'.format(rank, windows[rank].shape[0])

        # print 'Shape of windows: {}'.format(windows.shape)

    else:
        # Only need to do this on one process
        windows = None

    comm.Barrier()
    # Initialize the windows for a specific process
    # my_wins = np.zeros(shape=[windows[mpi_rank].shape[0], h5_pos.shape[1]], dtype=h5_pos.dtype)

    # print "Pre-Scatter \t my rank: {},\tmy_wins: {}".format(mpi_rank, my_wins)

    # Send each process it's set of windows
    my_wins = comm.scatter(windows, root=0)

    # print "Post-scatter \t my rank: {},\tmy_wins: {}".format(mpi_rank, my_wins)

    # Turn array of starting positions into a list of slices
    my_slices = [[slice(x, x + win_x), slice(y, y + win_y)] for x, y in my_wins]

    comm.Barrier()

    n_my_wins = len(my_wins)

    '''
    Create a matrix to add when counting.
    h5_V is usually small so go ahead and take S.V
    '''
    ds_V = np.dot(np.diag(h5_S[comp_slice]), h5_V[comp_slice, :]['Image Data']).T
    num_comps = len(h5_S[comp_slice])

    '''
    Initialize arrays to hold summed windows and counts for each position
    '''
    ones = np.ones([win_x, win_y, num_comps], dtype=np.uint32)
    my_counts = np.zeros([im_x, im_y, num_comps], dtype=np.uint32)
    my_clean_image = np.zeros([im_x, im_y, num_comps], dtype=np.float32)
    counts = np.zeros([im_x, im_y, num_comps], dtype=np.uint32)
    clean_image = np.zeros([im_x, im_y, num_comps], dtype=np.float32)

    '''
    Calculate the size of a given batch that will fit in the available memory
    '''
    mem_per_win = ds_V.itemsize * (ds_V.size + num_comps)

    max_memory = 4 * 1024 ** 3
    free_mem = max_memory - ds_V.size * ds_V.itemsize * mpi_size
    my_batch_size = int(free_mem / mem_per_win / mpi_size)
    my_start = 0

    if mpi_size > 1:
        if mpi_rank == 0:
            # print 'max memory', max_memory
            # print 'memory used per window', mem_per_win
            # print 'free memory', free_mem
            # print 'batch size per process', my_batch_size
    
            comm.isend(n_my_wins, dest=1)
    
        for rank in range(1, mpi_size-1):
            if mpi_rank == rank:
                my_start += comm.recv(source=mpi_rank - 1)
                comm.isend(n_my_wins+my_start, dest=mpi_rank + 1)
            comm.Barrier()
    
        if mpi_rank == mpi_size-1:
            my_start += comm.recv(source=mpi_rank - 1)
    
    comm.Barrier()

    print "my rank: {}\tmy start: {}".format(mpi_rank, my_start)

    if my_batch_size < 1:
        raise MemoryError('Not enough memory to perform Image Cleaning.')

    batch_slices = gen_batches(n_my_wins, my_batch_size, my_start)
    batch_win_slices = gen_batches(n_my_wins, my_batch_size)

    comm.Barrier()
    '''
    Loop over all batches.  Increment counts for window positions and
    add current window to total.
    '''
    for ibatch, (my_batch, my_win_batch) in enumerate(zip(batch_slices, batch_win_slices)):
        # print "my rank: {}\tmy batch: {}\tmy win batch: {}".format(mpi_rank, my_batch, my_win_batch)
        batch_wins = np.reshape(h5_U[my_batch, comp_slice][:, None, :] * ds_V[None, :, :], [-1, win_x, win_y, num_comps])
        # print mpi_rank, batch_wins.shape, len(my_slices)
        for islice, this_slice in enumerate(my_slices[my_win_batch]):
            iwin = ibatch * my_batch_size + islice + my_start
            if iwin % np.rint(n_my_wins / 10) == 0:
                per_done = np.rint(100 * iwin / n_my_wins)
                print('Process {} Reconstructing Image...{}% -- step # {} -- window {} -- slice {}'.format(mpi_rank,
                                                                                                           per_done,
                                                                                                           islice,
                                                                                                           iwin,
                                                                                                           this_slice))
            # if mpi_rank == 1:
            #     print "my rank: {}\tislice: {}\tthis_slice: {}\t iwin: {}".format(mpi_rank, islice, this_slice, iwin)
            my_counts[this_slice] += ones
            # print "my rank: {}\tthis_slice: {}\tislice: {}\tmy_start: {}".format(mpi_rank,
            #                                                                      this_slice,
            #                                                                      islice,
            #                                                                      my_start)
            # print "my image: {}\tbatch_wins: {}".format(my_clean_image.shape, batch_wins.shape)
            my_clean_image[this_slice] += batch_wins[islice]

        del batch_wins

    comm.Barrier()

    if mpi_rank == 0:
        print 'Finished summing chunks of windows on all processors.  Combining them with Allreduce.'


    comm.Allreduce(my_counts, counts, op=MPI.SUM)
    comm.Allreduce(my_clean_image, clean_image, op=MPI.SUM)

    comm.Barrier()

    # for rank in range(mpi_size):
    #     if mpi_rank == rank:
    #         my_clean_image.dump('image_array_{}'.format(mpi_rank))
    #         my_counts.dump('counts_array_{}'.format(mpi_rank))
    #         my_clean_image /= my_counts
    #         my_clean_image[np.isnan(my_clean_image)] = 0
    #         plt.imsave('counts_{}.png'.format(mpi_rank), np.sum(my_counts, axis=2))
    #         plt.imsave('image_{}.png'.format(mpi_rank), np.sum(my_clean_image, axis=2))
    #     else:
    #         pass
    #     comm.Barrier()

    # print 'my rank: {},\t my_counts=0: {}'.format(mpi_rank, np.argwhere(my_counts==0))
    # print 'my rank: {},\t counts=0: {}'.format(mpi_rank, np.argwhere(counts==0))
    #
    # print 'my rank: {},\t my_counts: {}'.format(mpi_rank, my_counts)
    # print 'my rank: {},\t counts: {}'.format(mpi_rank, counts)
    #
    # print 'my rank: {},\t my_clean_image: {}'.format(mpi_rank, my_clean_image)
    # print 'my rank: {},\t clean_image: {}'.format(mpi_rank, clean_image)

    del my_counts, my_clean_image, ds_V

    clean_image /= counts
    del counts
    '''
    Replace any NaNs with zeroes
    '''
    clean_image[np.isnan(clean_image)] = 0

    # if mpi_rank == 0:
    #     print 'Plotting final image.'
    #     clean_image.dump('image_array_final')
    #     plt.imsave('image_total.png', np.sum(clean_image, axis=2))

    comm.Barrier()

    '''
    Write the results to the file
    '''
    if mpi_rank == 0:
        print 'Creating new group'

    try:
        h5_clean_grp = h5_svd.create_group('PCA-Cleaned_Image_000')
    except ValueError:
        h5_clean_grp = h5_svd['PCA-Cleaned_Image_000']
        h5_clean_grp.clear()
    except:
        raise

    clean_chunking = calc_chunks([im_x * im_y, num_comps],
                                 clean_image.dtype.itemsize)
    if mpi_rank == 0:
        print im_x * im_y, num_comps
        print 'Image chunking {}'.format(clean_chunking)
    comm.Barrier()

    '''
    Create the datasets
    '''
    if mpi_rank == 0:
        print 'Creating Cleaned_Image dataset.'

    h5_clean = h5_clean_grp.create_dataset('Cleaned_Image',
                                           shape=(im_x * im_y, num_comps),
                                           chunks=clean_chunking,
                                           dtype=np.float32)

    comm.Barrier()
    h5_file.flush()
    comm.Barrier()

    '''
    Copy Position and Spectroscopic attributes from Raw_Image
    '''
    h5_clean.attrs['Position_Indices'] = h5_source_image.attrs['Position_Indices']
    h5_clean.attrs['Position_Values'] = h5_source_image.attrs['Position_Values']
    h5_clean.attrs['Spectroscopic_Indices'] = h5_U.attrs['Spectroscopic_Indices']
    h5_clean.attrs['Spectroscopic_Values'] = h5_U.attrs['Spectroscopic_Values']

    '''
    Write the components
    '''
    h5_clean_grp.attrs['components_used'] = np.arange(h5_S.size)[comp_slice]

    write_batches = [batch for batch in gen_batches_mpi(im_x*im_y, mpi_size)]

    if mpi_rank == 0:
        print 'Writing image to dataset'

    clean_image = clean_image.reshape(h5_clean.shape)

    comm.Barrier()
    print 'my rank: {}, I write: {}'.format(mpi_rank, write_batches[mpi_rank])
    h5_clean[write_batches[mpi_rank], :] = clean_image[write_batches[mpi_rank], :]

    if mpi_rank == 0:
        print 'image written'

    comm.Barrier()
    h5_file.flush()

    return h5_clean


def get_component_slice(components):
    """
    Check the components object to determine how to use it to slice the dataset

    Parameters
    ----------
    components : {int, iterable of ints, slice, or None}
        Input Options
        integer: Components less than the input will be kept
        length 2 iterable of integers: Integers define start and stop of component slice to retain
        other iterable of integers or slice: Selection of component indices to retain
        None: All components will be used
    Returns
    -------
    comp_slice : slice or numpy array of uints
        Slice or array specifying which components should be kept
    """

    comp_slice = slice(None)

    if isinstance(components, int):
        # Component is integer
        comp_slice = slice(0, components)
    elif hasattr(components, '__iter__') and not isinstance(components, dict):
        # Component is array, list, or tuple
        if len(components) == 2:
            # If only 2 numbers are given, use them as the start and stop of a slice
            comp_slice = slice(int(components[0]), int(components[1]))
        else:
            #Convert components to an unsigned integer array
            comp_slice = np.uint(np.round(components))
    elif isinstance(components, slice):
        # Components is already a slice
        comp_slice = components
    elif components is not None:
        raise TypeError('Unsupported component type supplied to clean_and_build.  Allowed types are integer, numpy array, list, tuple, and slice.')

    return comp_slice

if __name__ == '__main__':
    
    h5_path = None
    dataset_path = None
    retained_components = None

    comm = MPI.COMM_WORLD

    args, extra = getopt.getopt(sys.argv[1:], 'hi:d:c:')
    
    for arg,argval in args:
        '''
        Print a simple help
        '''
        print 'arg:',arg,
        print 'argval:',argval
        
        if arg == '-h':
            print 'clean_windows.py -i <inputfile> -d <path-to-svd-group> [-c <components-to-keep>]'
            sys.exit()
        elif arg == '-i':
            h5_path = argval
        elif arg == '-d':
            dataset_path = argval
        elif arg == '-c':
            retained_components = np.int(argval)
        else:
            print 'Unknown argument, value pair returned.'
            print 'This should never happen.'
            raise NotImplementedError('Unknown argument, value pair.')        
    
    '''
    Read any extra arguments into unassigned inputs in order
    '''    
    for arg in extra:
        if h5_path is None:
            h5_path = arg
        elif dataset_path is None:
            dataset_path = arg
        elif retained_components is None:
            retained_components = arg
        
    '''
    Get the file and dataset from the paths
    '''
    h5_path = os.path.abspath(h5_path)

    h5_file = h5py.File(h5_path, 'r+', driver='mpio', comm=MPI.COMM_WORLD)
    h5_file.atomic = True

    h5_svd = h5_file[dataset_path]

    comm.Barrier()

    h5_clean = clean_and_build_components(h5_svd, comm, retained_components)

    comm.Barrier()

    if comm.rank == 0:
        print 'Image has been successfully cleaned and rebuilt.'
        print 'Cleaned Image dataset located at {}'.format(h5_clean.name)

    comm.Barrier()
    h5_file.close()
    comm.Barrier()
    if comm.rank == 0:
        print 'File closed'
