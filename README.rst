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
1. Dask
2. mpi4py + existing Joblib
3. Pure mpi4py
4. pySpark
5. Workflows such as Swift

2. mpi4py+joblib
----------------
Strategies
~~~~~~~~~~
#. **1 rank / node**: Use an MPI + OpenMP paradigm where each rank is in charge of one node and computes via ``joblib`` within the node just as in pyUSID / pycroscopy.

   Pros:

   * Easy to understand and implement since each node continues to do whatever a laptop would do / has been doing

   Cons:

   * ``joblib`` sometimes does not like to work with ``numpy`` and ``mpi4py``
   * Requires that the PBS script + ``mpiexec`` command be clear about assigning only one rank / node

#. **Pure MPI**: Use one rank per logical core. Only one rank within each node will write the data meaning that ``send`` and ``gather`` MPI commands will be necessary
   for the ranks within a node to collect results before writing.

   Pros:

   * Circumvents ``joblib`` related problems since it obviates the need for ``joblib``
   * (Potentially) less restrictive PBS script

   Cons:

   * Slightly more complicated in that additional book-keeping would be required for the relationships (master) within each node
   * If a node has fewer ranks than the number of logical cores, those cores are wasted.
   * The rank that collects all the results may not have sufficient memory. This may limit how much each rank can compute at a given time

#. **Arbitrary MPI ranks / node**: Use a combination of joblib and MPI and pose no restrictions whatsoever on the number of ranks or configuration

   Pros:

   * Probably the programmatically "proper" way to do this
   * PBS script and ``mpiexec`` call can be configured in any way

   Cons:

   * Has nearly all the major cons of the two above approaches
   * ``joblib`` sometimes does not like to work with ``numpy`` and ``mpi4py``
   * Noticeably more complicated in that additional book-keeping would be required for the relationships (master) within each node
   * The rank that collects all the results may not have sufficient memory. This may limit how much each rank can compute at a given time

Status
~~~~~~
Only the first of the three mpi4py+joblib approaches has been explored so far
#. ``Process`` class requires no more changes for **basic** MPI functionality / scaling embarrassingly parallel problems

   * Checkpointing has not yet been implemented (ran out of allocation time for example). Challenges:

     * Need to figure out how to interrupt / checkpoint without corrupting hdf5 file
     * Need to figure out metadata that will be necessary to indicate completed pixels

       * One solution - ``last_pixel`` = list of completed slices.
       * ``Process.__init__`` should build a list of pixels that need to be computed and distribute those to ranks via ``scatter`` instead.
#. Image Cleaning: Already tested and working code by Chris
#. Signal Filter - now working just fine
#. GIV Bayesian inference - `problem with joblib + MPI <./giv_bayesian/bayesian_script_mpi.py>`_. Works fine in serial processing mode.

Observations
~~~~~~~~~~~~
* Minimal changes are required for the children classes of ``pyUSID.Process`` - mainly in verbose print statements
* First test the dataset creation step with the computation disabled to speed up debugging time. Most of the challenges are in the dataset creation portion.
* ``h5py`` (parallel) results in **segmentation faults** for the following situations:

  * If ``compression`` is specified when creating datasets. Known issue with no workaround
  * ``if rank == 0: write_simple_attrs(....)`` <-- Make all ranks write attributes
