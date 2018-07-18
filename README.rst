mpiUSID
=======

MPI versions of the embarrassingly-parallel ``Process`` classes in pycroscopy and pyUSID.

* The emphasis is to develop the ``pyUSID.Process`` class such that it continues to work for laptops but also works on HPC with minimal modifications to children classes.
  Once certified to work well for a handful of examples, the changes in the class(es) will be rolled back into ``pyUSID`` and even ``pycroscopy``.
* Current strategy is to use an MPI + OpenMP paradigm instead of a pure MPI paradigm - We don't want too many ranks writing to the HDF5 file.
  Ideally, limit to **1 rank / node**. Each rank is in charge of one node and computes via ``joblib`` within the node just as in pyUSID / pycroscopy.
* Code here is developed and tested on ORNL CADES SHPC OR Condo only for the time being. The code should, in theory, be portable to OLCF or other machines.

Status
------
#. ``Process`` class requires no more changes for **basic** MPI functionality / scaling embarrassingly parallel problems
#. Image Cleaning: Already tested and working code by Chris
#. Signal Filter - now working just fine
#. GIV Bayesian inference - problem with joblib. Works fine in serial processing mode.
#. Checkpointing has not yet been implemented (ran out of allocation time for example). Challenges:

   * Need to figure out how to interrupt / checkpoint without corrupting hdf5 file
   * Need to figure out metadata that will be necessary to indicate completed pixels

     * One solution - ``last_pixel`` = list of completed slices.
     * ``Process.__init__`` should build a list of pixels that need to be computed and distribute those to ranks via ``scatter`` instead.

Observations
------------
* Minimal changes are required for the children classes of ``pyUSID.Process`` - mainly in verbose print statements
* First test the dataset creation step with the computation disabled to speed up debugging time. Most of the challenges are in the dataset creation portion.
* ``h5py`` (parallel) results in **segmentation faults** for the following situations:

  * If ``compression`` is specified when creating datasets. Known issue with no workaround
  * ``if rank == 0: write_simple_attrs(....)`` <-- Make all ranks write attributes
