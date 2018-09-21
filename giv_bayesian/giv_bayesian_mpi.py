# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 11:48:53 2017

@author: Suhas Somnath

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from pyUSID.io.dtype_utils import stack_real_to_compound
from pyUSID.io.hdf_utils import write_main_dataset, create_results_group, write_simple_attrs, \
    print_tree, get_attributes
from pyUSID.io.write_utils import Dimension
from pyUSID import USIDataset

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        # mpi4py available but NOT called via mpirun or mpiexec => single node
        MPI = None
except ImportError:
    # mpi4py not even present! Single node by default:
    MPI = None

from mpi_process import Process, parallel_compute
from giv_utils import do_bayesian_inference, bayesian_inference_on_period

cap_dtype = np.dtype({'names': ['Forward', 'Reverse'],
                      'formats': [np.float32, np.float32]})
# TODO : Take lesser used bayesian inference params from kwargs if provided


def create_empty_dataset(source_dset, dtype, dset_name, h5_group=None, new_attrs=None, skip_refs=False):
    """
    Creates an empty dataset in the h5 file based on the provided dataset in the same or specified group
    Parameters
    ----------
    source_dset : h5py.Dataset object
        Source object that provides information on the group and shape of the dataset
    dtype : dtype
        Data type of the fit / guess datasets
    dset_name : String / Unicode
        Name of the dataset
    h5_group : h5py.Group object, optional. Default = None
        Group within which this dataset will be created
    new_attrs : dictionary (Optional)
        Any new attributes that need to be written to the dataset
    skip_refs : boolean, optional
        Should ObjectReferences and RegionReferences be skipped when copying attributes from the
        `source_dset`
    Returns
    -------
    h5_new_dset : h5py.Dataset object
        Newly created dataset
    """
    import h5py
    from pyUSID.io.dtype_utils import validate_dtype
    from pyUSID.io.hdf_utils import copy_attributes, check_if_main, write_book_keeping_attrs
    from pyUSID import USIDataset
    import sys
    if sys.version_info.major == 3:
        unicode = str

    if not isinstance(source_dset, h5py.Dataset):
        raise TypeError('source_deset should be a h5py.Dataset object')
    _ = validate_dtype(dtype)
    if new_attrs is not None:
        if not isinstance(new_attrs, dict):
            raise TypeError('new_attrs should be a dictionary')
    else:
        new_attrs = dict()

    if h5_group is None:
        h5_group = source_dset.parent
    else:
        if not isinstance(h5_group, (h5py.Group, h5py.File)):
            raise TypeError('h5_group should be a h5py.Group or h5py.File object')

    if not isinstance(dset_name, (str, unicode)):
        raise TypeError('dset_name should be a string')
    dset_name = dset_name.strip()
    if len(dset_name) == 0:
        raise ValueError('dset_name cannot be empty!')
    if '-' in dset_name:
        warn('dset_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(dset_name, dset_name.replace('-', '_')))
    dset_name = dset_name.replace('-', '_')

    if dset_name in h5_group.keys():
        if isinstance(h5_group[dset_name], h5py.Dataset):
            warn('A dataset named: {} already exists in group: {}'.format(dset_name, h5_group.name))
            h5_new_dset = h5_group[dset_name]
            # Make sure it has the correct shape and dtype
            if any((source_dset.shape != h5_new_dset.shape, dtype != h5_new_dset.dtype)):
                warn('Either the shape (existing: {} desired: {}) or dtype (existing: {} desired: {}) of the dataset '
                     'did not match with expectations. Deleting and creating a new one.'.format(h5_new_dset.shape,
                                                                                                source_dset.shape,
                                                                                                h5_new_dset.dtype,
                                                                                                dtype))
                del h5_new_dset, h5_group[dset_name]
                h5_new_dset = h5_group.create_dataset(dset_name, shape=source_dset.shape, dtype=dtype,
                                                      chunks=source_dset.chunks)
        else:
            raise KeyError('{} is already a {} in group: {}'.format(dset_name, type(h5_group[dset_name]),
                                                                    h5_group.name))

    else:
        h5_new_dset = h5_group.create_dataset(dset_name, shape=source_dset.shape, dtype=dtype,
                                              chunks=source_dset.chunks)

    # This should link the ancillary datasets correctly
    h5_new_dset = copy_attributes(source_dset, h5_new_dset, skip_refs=skip_refs)
    h5_new_dset.attrs.update(new_attrs)

    if check_if_main(h5_new_dset):
        h5_new_dset = USIDataset(h5_new_dset)
        # update book keeping attributes
        write_book_keeping_attrs(h5_new_dset)

    return h5_new_dset


class GIVBayesian(Process):

    def __init__(self, h5_main, ex_freq, gain, num_x_steps=250, r_extra=110, **kwargs):
        """
        Applies Bayesian Inference to General Mode IV (G-IV) data to extract the true current

        Parameters
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        ex_freq : float
            Frequency of the excitation waveform
        gain : uint
            Gain setting on current amplifier (typically 7-9)
        num_x_steps : uint (Optional, default = 250)
            Number of steps for the inferred results. Note: this may be end up being slightly different from specified.
        r_extra : float (Optional, default = 110 [Ohms])
            Extra resistance in the RC circuit that will provide correct current and resistance values
        kwargs : dict
            Other parameters specific to the Process class and nuanced bayesian_inference parameters
        """
        super(GIVBayesian, self).__init__(h5_main, **kwargs)
        self.gain = gain
        self.ex_freq = ex_freq
        self.r_extra = r_extra
        self.num_x_steps = int(num_x_steps)
        if self.num_x_steps % 4 == 0:
            self.num_x_steps = ((self.num_x_steps // 2) + 1) * 2
        if self.verbose and self.mpi_rank == 0:
            print('ensuring that half steps should be odd, num_x_steps is now', self.num_x_steps)

        self.h5_main = USIDataset(self.h5_main)

        # take these from kwargs
        bayesian_parms = {'gam': 0.03, 'e': 10.0, 'sigma': 10.0, 'sigmaC': 1.0, 'num_samples': 2E3}

        self.parms_dict = {'freq': self.ex_freq, 'num_x_steps': self.num_x_steps, 'r_extra': self.r_extra}
        self.parms_dict.update(bayesian_parms)

        self.process_name = 'Bayesian_Inference'
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        # Should not be extracting excitation this way!
        h5_spec_vals = self.h5_main.h5_spec_vals[0]
        self.single_ao = np.squeeze(h5_spec_vals[()])

        roll_cyc_fract = -0.25
        self.roll_pts = int(self.single_ao.size * roll_cyc_fract)
        self.rolled_bias = np.roll(self.single_ao, self.roll_pts)

        dt = 1 / (ex_freq * self.single_ao.size)
        self.dvdt = np.diff(self.single_ao) / dt
        self.dvdt = np.append(self.dvdt, self.dvdt[-1])

        self.reverse_results = None
        self.forward_results = None
        self._bayes_parms = None

    def test(self, pix_ind=None, show_plots=True):
        """
        Tests the inference on a single pixel (randomly chosen unless manually specified) worth of data.

        Parameters
        ----------
        pix_ind : int, optional. default = random
            Index of the pixel whose data will be used for inference
        show_plots : bool, optional. default = True
            Whether or not to show plots

        Returns
        -------
        fig, axes
        """
        if self.mpi_rank > 0:
            return

        if pix_ind is None:
            pix_ind = np.random.randint(0, high=self.h5_main.shape[0])
        other_params = self.parms_dict.copy()
        # removing duplicates:
        _ = other_params.pop('freq')

        return bayesian_inference_on_period(self.h5_main[pix_ind], self.single_ao, self.parms_dict['freq'],
                                            show_plots=show_plots, **other_params)

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
        super(GIVBayesian, self)._set_memory_and_cores(cores=cores, mem=mem)
        # Remember that the default number of pixels corresponds to only the raw data that can be held in memory
        # In the case of simplified Bayesian inference, four (roughly) equally sized datasets need to be held in memory:
        # raw, compensated current, resistance, variance
        self._max_pos_per_read = self._max_pos_per_read // 4  # Integer division
        # Since these computations take far longer than functional fitting, do in smaller batches:
        self._max_pos_per_read = min(100, self._max_pos_per_read)

        if self.verbose and self.mpi_rank == 0:
            print('Max positions per read set to {}'.format(self._max_pos_per_read))

    def _create_results_datasets(self):
        """
        Creates hdf5 datasets and datagroups to hold the resutls
        """
        # create all h5 datasets here:
        num_pos = self.h5_main.shape[0]

        if self.verbose and self.mpi_rank == 0:
            print('Now creating the datasets')

        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

        write_simple_attrs(self.h5_results_grp, {'algorithm_author': 'Kody J. Law', 'last_pixel': 0})
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        if self.verbose and self.mpi_rank == 0:
            print('created group: {} with attributes:'.format(self.h5_results_grp.name))
            print(get_attributes(self.h5_results_grp))

        # One of those rare instances when the result is exactly the same as the source
        self.h5_i_corrected = create_empty_dataset(self.h5_main, np.float32, 'Corrected_Current', h5_group=self.h5_results_grp)

        if self.verbose and self.mpi_rank == 0:
            print('Created I Corrected')
            # print_tree(self.h5_results_grp)

        # For some reason, we cannot specify chunks or compression!
        # The resistance dataset requires the creation of a new spectroscopic dimension
        self.h5_resistance = write_main_dataset(self.h5_results_grp, (num_pos, self.num_x_steps), 'Resistance', 'Resistance',
                                                'GOhms', None, Dimension('Bias', 'V', self.num_x_steps),
                                                dtype=np.float32, # chunks=(1, self.num_x_steps), #compression='gzip',
                                                h5_pos_inds=self.h5_main.h5_pos_inds,
                                                h5_pos_vals=self.h5_main.h5_pos_vals)

        if self.verbose and self.mpi_rank == 0:
            print('Created Resistance')
            # print_tree(self.h5_results_grp)

        assert isinstance(self.h5_resistance, USIDataset)  # only here for PyCharm
        self.h5_new_spec_vals = self.h5_resistance.h5_spec_vals

        # The variance is identical to the resistance dataset
        self.h5_variance = create_empty_dataset(self.h5_resistance, np.float32, 'R_variance')

        if self.verbose and self.mpi_rank == 0:
            print('Created Variance')
            # print_tree(self.h5_results_grp)

        # The capacitance dataset requires new spectroscopic dimensions as well
        self.h5_cap = write_main_dataset(self.h5_results_grp, (num_pos, 1), 'Capacitance', 'Capacitance', 'pF', None,
                                         Dimension('Direction', '', [1]),  h5_pos_inds=self.h5_main.h5_pos_inds,
                                         h5_pos_vals=self.h5_main.h5_pos_vals, dtype=cap_dtype, #compression='gzip',
                                         aux_spec_prefix='Cap_Spec_')

        if self.verbose and self.mpi_rank == 0:
            print('Created Capacitance')
            # print_tree(self.h5_results_grp)
            print('Done creating all results datasets!')

        if self.mpi_size > 1:
            self.mpi_comm.Barrier()
        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
        self.h5_cap = self.h5_results_grp['Capacitance']
        self.h5_variance = self.h5_results_grp['R_variance']
        self.h5_resistance = self.h5_results_grp['Resistance']
        self.h5_i_corrected = self.h5_results_grp['Corrected_Current']

    def _write_results_chunk(self):
        """
        Writes data chunks back to the h5 file
        """

        if self.verbose:
            print('Rank {} - Started accumulating results for this chunk'.format(self.mpi_rank))

        num_pixels = len(self.forward_results)
        cap_mat = np.zeros((num_pixels, 2), dtype=np.float32)
        r_inf_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        r_var_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        i_cor_sin_mat = np.zeros((num_pixels, self.single_ao.size), dtype=np.float32)

        for pix_ind, i_meas, forw_results, rev_results in zip(range(num_pixels), self.data,
                                                              self.forward_results, self.reverse_results):
            full_results = dict()
            for item in ['cValue']:
                full_results[item] = np.hstack((forw_results[item], rev_results[item]))
                # print(item, full_results[item].shape)

            # Capacitance is always doubled - halve it now (locally):
            # full_results['cValue'] *= 0.5
            cap_val = np.mean(full_results['cValue']) * 0.5

            # Compensating the resistance..
            """
            omega = 2 * np.pi * self.ex_freq
            i_cap = cap_val * omega * self.rolled_bias
            """
            i_cap = cap_val * self.dvdt
            i_extra = self.r_extra * 2 * cap_val * self.single_ao
            i_corr_sine = i_meas - i_cap - i_extra

            # Equivalent to flipping the X:
            rev_results['x'] *= -1

            # Stacking the results - no flipping required for reverse:
            for item in ['x', 'mR', 'vR']:
                full_results[item] = np.hstack((forw_results[item], rev_results[item]))

            i_cor_sin_mat[pix_ind] = i_corr_sine
            cap_mat[pix_ind] = full_results['cValue'] * 1000  # convert from nF to pF
            r_inf_mat[pix_ind] = full_results['mR']
            r_var_mat[pix_ind] = full_results['vR']

        # Now write to h5 files:
        if self.verbose:
            print('Rank {} - Finished accumulating results. Writing results of chunk to h5'.format(self.mpi_rank))

        if self._start_pos == 0:
            self.h5_new_spec_vals[0, :] = full_results['x']  # Technically this needs to only be done once

        pos_slice = slice(self._start_pos, self._end_pos)
        self.h5_cap[pos_slice] = np.atleast_2d(stack_real_to_compound(cap_mat, cap_dtype)).T
        self.h5_variance[pos_slice] = r_var_mat
        self.h5_resistance[pos_slice] = r_inf_mat
        self.h5_i_corrected[pos_slice] = i_cor_sin_mat

        # Leaving in this provision that will allow restarting of processes
        self.h5_results_grp.attrs['last_pixel'] = self._end_pos

        # Disabling flush because h5py-parallel doesn't like it
        # self.h5_main.file.flush()

        print('Rank {} - Finished processing up to pixel {} of {}'
              '.'.format(self.mpi_rank, self._end_pos, self._rank_end_pos))

        # Now update the start position
        self._start_pos = self._end_pos

    def _unit_computation(self, *args, **kwargs):
        """
        Processing per chunk of the dataset

        Parameters
        ----------
        args : list
            Not used
        kwargs : dictionary
            Not used
        """
        half_v_steps = self.single_ao.size // 2

        # first roll the data
        rolled_raw_data = np.roll(self.data, self.roll_pts, axis=1)
        # Ensure that the bias has a positive slope. Multiply current by -1 accordingly
        if self.verbose:
            print('Rank {} beginning parallel compute for Forward'.format(self.mpi_rank))
        self.reverse_results = parallel_compute(rolled_raw_data[:, :half_v_steps] * -1, do_bayesian_inference,
                                                cores=self._cores,
                                                func_args=[self.rolled_bias[:half_v_steps] * -1, self.ex_freq],
                                                func_kwargs=self._bayes_parms, lengthy_computation=True,
                                                verbose=self.verbose)

        if self.verbose:
            print('Rank {} finished processing forward sections. Now working on reverse sections....'.format(self.mpi_rank))

        self.forward_results = parallel_compute(rolled_raw_data[:, half_v_steps:], do_bayesian_inference,
                                                cores=self._cores,
                                                func_args=[self.rolled_bias[half_v_steps:], self.ex_freq],
                                                func_kwargs=self._bayes_parms, lengthy_computation=True,
                                                verbose=self.verbose)
        if self.verbose:
            print('Rank {} Finished processing reverse loops (and this chunk)'.format(self.mpi_rank))

    def compute(self, override=False, *args, **kwargs):
        """
        Creates placeholders for the results, applies the inference to the data, and writes the output to the file.
        Consider calling test() before this function to make sure that the parameters are appropriate.

        Parameters
        ----------
        override : bool, optional. default = False
            By default, compute will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.
        args : list
            Not used
        kwargs : dictionary
            Not used

        Returns
        -------
        h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results
        """

        # remove additional parm and halve the x points
        self._bayes_parms = self.parms_dict.copy()
        self._bayes_parms['num_x_steps'] = self.num_x_steps // 2
        self._bayes_parms['econ'] = True
        del(self._bayes_parms['freq'])

        return super(GIVBayesian, self).compute(override=override, *args, **kwargs)
