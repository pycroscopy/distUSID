#!/bin/bash

### Set the job name. Your output files will share this name.
#PBS -N mpiGivBayesian
### Enter your email address. Errors will be emailed to this address.
#PBS -M somnaths@ornl.gov
### Node spec, number of nodes and processors per node that you desire.
### One node and 36 cores per node in this case.
#PBS -l nodes=2:ppn=36
### Tell PBS the anticipated runtime for your job, where walltime=HH:MM:S.
#PBS -l walltime=0:00:15:0
### The LDAP group list they need; cades-birthright in this case.
#PBS -W group_list=cades-ccsd
### Your account type. Birthright in this case.
#PBS -A ccsd
### Quality of service set to burst.
#PBS -l qos=std


## main program ##

### Remove old modules to ensure a clean state.
module purge

### Load modules (your programming environment)
# Normally would use PE-gnu but Ketan has compiled numpy to use MKL on Condo
module load PE-intel
### Load custom python virtual environment
module load python/3.6.3

### Check loaded modules
module list

### Forcing MKL to use 1 thread only:
# MUST KEEP THESE LINES. WILL NOT WORK OTHERWISE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

### Switch to the working directory (path of your PBS script).
EGNAME=giv_bayesian_pure_mpi
DATA_PATH=$HOME/giv/pzt_nanocap_6_just_translation_filt_resh_copy.h5
SCRIPTS_PATH=$HOME/mpi_tutorials/$EGNAME
WORK_PATH=/lustre/or-hydra/cades-ccsd/syz/pycroscopy_ensemble

cd $WORK_PATH
mkdir $EGNAME
cd $EGNAME

### Show current directory.
pwd

### Copy data:
DATA_NAME=giv_raw.h5
rm -rf $DATA_NAME
cp $DATA_PATH $DATA_NAME

### Copy python files:
cp $SCRIPTS_PATH/mpi_process.py .
cp $SCRIPTS_PATH/giv_bayesian_mpi.py .
cp $SCRIPTS_PATH/giv_utils.py .
cp $SCRIPTS_PATH/bayesian_script_mpi.py .

# See the contents in Lustre to make sure they are OK
ls -hl

### MPI run followed by the name/path of the binary.
mpiexec --mca mpi_warn_on_fork 0 -use-hwthread-cpus python bayesian_script_mpi.py
### mpiexec -use-hwthread-cpus python -m cProfile -s cumtime bayesian_script_mpi.py