mpiUSID
=======

MPI versions of popular classes in pycroscopy and pyUSID. 
The emphasis is to develop the ``pyUSID.Process`` class such that it continues to work for laptops but also works on HPC with minimal modifications to children classes.
Once certified to work well for a handful of examples, the changes in the class(es) will be rolled back into ``pyUSID`` and even ``pycroscopy``.
Code here is tested and developed on CADES SHPC Condo only for the time being.

Observations
------------
* Process class more or less requires no more changes for basic functionality
* Minimal changes are required for the children classes of ``pyUSID.Process`` - mainly in verbose print statements
* ``h5py`` (parallel) results in **segmentation faults** for the following situations:

  * ``h5_file.flush()``
  * If ``chunks`` is specified when creating datasets
  * If ``compression`` is specified when creating datasets
  * ``h5_file.close()`` in the case of ``SignalFilter``

* ``joblib`` appears to hang for bayesian inference. Works fine in serial processing mode.
* Checkpointing has not yet been implemented (ran out of allocation time for example). Challenges:

  * Need to figure out how to interrupt / checkpoint without corrupting hdf5 file
  * Need to figure out metadata that will be necessary to indicate completed pixels

    * One solution - ``last_pixel`` = list of completed slices.
    * ``Process.__init__`` should build a list of pixels that need to be computed and distribute those to ranks via ``scatter`` instead.
