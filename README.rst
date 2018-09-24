distUSID
=======

Distributed versions of the embarrassingly-parallel ``Process`` classes in pycroscopy and pyUSID.

* The emphasis is to develop the ``pyUSID.Process`` class such that it continues to work for laptops but also works on HPC / cloud clusters with minimal modifications to children classes (in pycroscopy).
  Once certified to work well for a handful of examples, the changes in the class(es) will be rolled back into ``pyUSID`` and even ``pycroscopy``.
* Code here is developed and tested on ORNL CADES SHPC OR Condo only for the time being. The code should, in theory, be portable to OLCF or other machines.

Limitations
-----------
Before diving in to the many strategies for going about how to solve this problem, it is important to be cognizant of the restrictions:

* The user-facing sub-classes should see minimal changes to enable distributed computing and not require expert knowledge of MPI or other paradigms
* There is a finite bandwidth available for file I/O even on high-performance-computing resources. This means that the number of parallel file writing
  operations should be minimized. Even with a single node with 36 cores, we do not want 36 processes / ranks waiting on each other to write data to the file.
  The majority of the time should be spent on the computation which is the main problem that has necessitated distributed computing.

Strategies
----------
#. Dask
#. Pure mpi4py
#. mpi4py + existing Joblib
#. pySpark
#. Workflows such as Swift

2. Pure mpi4py
--------------
Use one rank per logical core. All ranks read and write to the file. Code available on the `pure_mpi <https://github.com/pycroscopy/distUSID/tree/pure_mpi>`_ branch

**Pros**:

* Circumvents ``joblib`` related problems since it obviates the need for ``joblib``
* (Potentially) less restrictive PBS script

**Cons**:

* If a node has fewer ranks than the number of logical cores, those cores are wasted. This minor problem can be fixed

Status
~~~~~~
* Works very well for both the ``SignalFilter`` and the ``GIVBayesian`` class in addition to Chris' success on the ``image windowing``
* This same code **had** been `generalized <https://github.com/pycroscopy/distUSID/commit/4e4e367230c9a85540828b7d8e56cc261f135fae>`_
  to capture the two sub-cases of mpi4py+joblib below . However, this causes ``GIVBayesian`` to fail - just does not compute anything at all. No errors observed.

  * If a fix is discovered, this capability can be enabled with just `2 lines <https://github.com/pycroscopy/distUSID/commit/3d43614e8bd1ae722c26e72d7d1a95dbeac4cee8>`_.
  * This may be related to some complication in the `math libraries <https://github.com/pycroscopy/distUSID/commit/3930df86c6119226702628145090726ad1f00312>`_
* Have not yet seen any problems with regards to the bottleneck on up to 4 nodes (36 cores each). Benchmarking will be necessary for identify bottlenecks
* Comprehensive checkpointing / resuming capability has also been incorporated within the ``Process`` class
* The ``Process`` class has been made even more robust against accidental damage from user-side by moving more underlying code into private variables.
* Minimal changes are required for the children classes of ``pyUSID.Process``:

  * mainly in verbose print statements - need to check for ``rank == 0``
  * ``Process`` completely handles all check-pointing (legacy + new) and flushing the file after each batch. The user-side code literally only needs to write to the HDF5 datasets

* Plenty of documentation about the thought process included within the ``Process`` class file.
* The ``Process`` class from this branch will be rolled into pyUSID after some checks

Tips and Gotchas
~~~~~~~~~~~~~~~~
* First test the dataset creation step with the computation disabled to speed up debugging time. Most of the challenges are in the dataset creation portion.
* ``h5py`` (parallel) results in **segmentation faults** for the following situations:

  * If ``compression`` is specified when creating datasets. Known issue with no workaround
  * ``if rank == 0: write_simple_attrs(....)`` <-- Make all ranks write attributes
* Environment variables need to be set in the PBS script to minimize conflicts between LAPACK's preference to use threading and MPI / multiprocessing.
  Two `environment variables <https://github.com/pycroscopy/distUSID/commit/72d8ac086ee974a4ed644fbe55738d198b7265ec>`_ made a night-and-day difference
  in the `pure_mpi <https://github.com/pycroscopy/distUSID/tree/pure_mpi>`_ branch.

  * Setting these variables within ``parallel_compute()`` had the `same effect <https://github.com/pycroscopy/distUSID/commit/3ccdacfa32ac97af7eb9994a1562ea9c0caf51e5>`_ as not setting these environment variables at all.

3. mpi4py+joblib
----------------
#. **1 rank / node**: Use an MPI + OpenMP paradigm where each rank is in charge of one node and computes via ``joblib`` within the node just as in pyUSID / pycroscopy. See the `mpi_plus_joblib <https://github.com/pycroscopy/distUSID/tree/mpi_plus_joblib)>`_ branch

   **Pros**:

   * Easy to understand and implement since each node continues to do whatever a laptop would do / has been doing

   **Cons**:

   * ``joblib`` sometimes does not like to work with ``numpy`` and ``mpi4py``

   **Status**:

   * Worked for the ``SignalFilter`` but not for the ``GIVBayesian`` class.

#. **Arbitrary MPI ranks / node**: Use a combination of joblib and MPI and pose no restrictions whatsoever on the number of ranks or configuration

   **Pros**:

   * Probably the programmatically "proper" way to do this
   * PBS script and ``mpiexec`` call can be configured in any way

   **Cons**:

   * Has nearly all the major cons of the two above approaches
   * ``joblib`` sometimes does not like to work with ``numpy`` and ``mpi4py``
   * Noticeably more complicated in that additional book-keeping would be required for the relationships (master) within each node
   * The rank that collects all the results may not have sufficient memory. This may limit how much each rank can compute at a given time

   **Status**:

   * As mentioned above, the ``Process`` class in the `pure_mpi <https://github.com/pycroscopy/distUSID/tree/pure_mpi>`_ branch already
     captures this use-case but this refuses to work for ``GIVBayesian`` just like in the `mpi_plus_joblib <https://github.com/pycroscopy/distUSID/tree/mpi_plus_joblib)>`_ branch
