from __future__ import division, print_function, absolute_import
import sys
from collections import Iterable
from warnings import warn
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np
from fft import get_noise_floor, are_compatible_filters, build_composite_freq_filter
from pyUSID import USIDataset
from pyUSID.io.hdf_utils import check_if_main, get_attr, write_main_dataset, create_results_group
from pyUSID.viz.plot_utils import set_tick_font_size, plot_curves
from pyUSID.io.write_utils import Dimension

if sys.version_info.major == 3:
    unicode = str

# TODO: Phase rotation not implemented correctly. Find and use excitation frequency


###############################################################################

def test_filter(resp_wfm, frequency_filters=None, noise_threshold=None, excit_wfm=None, show_plots=True,
                plot_title=None, verbose=False):
    """
    Filters the provided response with the provided filters.
    Parameters
    ----------
    resp_wfm : array-like, 1D
        Raw response waveform in the time domain
    frequency_filters : (Optional) FrequencyFilter object or list of
        Frequency filters to apply to signal
    noise_threshold : (Optional) float
        Noise threshold to apply to signal
    excit_wfm : (Optional) 1D array-like
        Excitation waveform in the time domain. This waveform is necessary for plotting loops. If the length of
        resp_wfm matches excit_wfm, a single plot will be returned with the raw and filtered signals plotted against the
        excit_wfm. Else, resp_wfm and the filtered (filt_data) signal will be broken into chunks matching the length of
        excit_wfm and a figure with multiple plots (one for each chunk) with the raw and filtered signal chunks plotted
        against excit_wfm will be returned for fig_loops
    show_plots : (Optional) Boolean
        Whether or not to plot FFTs before and after filtering
    plot_title : str / unicode (Optional)
        Title for the raw vs filtered plots if requested. For example - 'Row 15'
    verbose : (Optional) Boolean
        Prints extra debugging information if True.  Default False
    Returns
    -------
    filt_data : 1D numpy float array
        Filtered signal in the time domain
    fig_fft : matplotlib.pyplot.figure object
        handle to the plotted figure if requested, else None
    fig_loops : matplotlib.pyplot.figure object
        handle to figure with the filtered signal and raw signal plotted against the excitation waveform
    """
    if not isinstance(resp_wfm, (np.ndarray, list)):
        raise TypeError('resp_wfm should be array-like')
    resp_wfm = np.array(resp_wfm)

    show_loops = False
    if excit_wfm is not None and show_plots:
        if len(resp_wfm) % len(excit_wfm) == 0:
            show_loops = True
        else:
            raise ValueError('Length of resp_wfm should be divisibe by length of excit_wfm')
    if show_loops:
        if plot_title is None:
            plot_title = 'FFT Filtering'
        else:
            assert isinstance(plot_title, (str, unicode))

    if frequency_filters is None and noise_threshold is None:
        raise ValueError('Need to specify at least some noise thresholding / frequency filter')

    if noise_threshold is not None:
        if noise_threshold >= 1 or noise_threshold <= 0:
            raise ValueError('Noise threshold must be within (0 1)')

    samp_rate = 1
    composite_filter = 1
    if frequency_filters is not None:
        if not isinstance(frequency_filters, Iterable):
            frequency_filters = [frequency_filters]
        if not are_compatible_filters(frequency_filters):
            raise ValueError('frequency filters must be a single or list of FrequencyFilter objects')
        composite_filter = build_composite_freq_filter(frequency_filters)
        samp_rate = frequency_filters[0].samp_rate

    resp_wfm = np.array(resp_wfm)
    num_pts = resp_wfm.size

    fft_pix_data = np.fft.fftshift(np.fft.fft(resp_wfm))

    if noise_threshold is not None:
        noise_floor = get_noise_floor(fft_pix_data, noise_threshold)[0]
        if verbose:
            print('The noise_floor is', noise_floor)

    fig_fft = None
    if show_plots:
        w_vec = np.linspace(-0.5 * samp_rate, 0.5 * samp_rate, num_pts) * 1E-3

        fig_fft, [ax_raw, ax_filt] = plt.subplots(figsize=(12, 8), nrows=2)
        axes_fft = [ax_raw, ax_filt]
        set_tick_font_size(axes_fft, 14)

        r_ind = num_pts
        if isinstance(composite_filter, np.ndarray):
            r_ind = np.max(np.where(composite_filter > 0)[0])

        x_lims = slice(len(w_vec) // 2, r_ind)
        amp = np.abs(fft_pix_data)
        ax_raw.semilogy(w_vec[x_lims], amp[x_lims], label='Raw')
        if frequency_filters is not None:
            ax_raw.semilogy(w_vec[x_lims], (composite_filter[x_lims] + np.min(amp)) * (np.max(amp) - np.min(amp)),
                            linewidth=3, color='orange', label='Composite Filter')
        if noise_threshold is not None:
            ax_raw.axhline(noise_floor,
                           # ax_raw.semilogy(w_vec, np.ones(r_ind - l_ind) * noise_floor,
                           linewidth=2, color='r', label='Noise Threshold')
        ax_raw.legend(loc='best', fontsize=14)
        ax_raw.set_title('Raw Signal', fontsize=16)
        ax_raw.set_ylabel('Magnitude (a. u.)', fontsize=14)

    fft_pix_data *= composite_filter

    if noise_threshold is not None:
        fft_pix_data[np.abs(fft_pix_data) < noise_floor] = 1E-16  # DON'T use 0 here. ipython kernel dies

    if show_plots:
        ax_filt.semilogy(w_vec[x_lims], np.abs(fft_pix_data)[x_lims])
        ax_filt.set_title('Filtered Signal', fontsize=16)
        ax_filt.set_xlabel('Frequency(kHz)', fontsize=14)
        ax_filt.set_ylabel('Magnitude (a. u.)', fontsize=14)
        if noise_threshold is not None:
            ax_filt.set_ylim(bottom=noise_floor)  # prevents the noise threshold from messing up plots
        fig_fft.tight_layout()

    filt_data = np.real(np.fft.ifft(np.fft.ifftshift(fft_pix_data)))

    if verbose:
        print('The shape of the filtered data is {}'.format(filt_data.shape))
        print('The shape of the excitation waveform is {}'.format(excit_wfm.shape))

    fig_loops = None
    if show_loops:
        if len(resp_wfm) == len(excit_wfm):
            # single plot:
            fig_loops, axis = plt.subplots(figsize=(5.5, 5))
            axis.plot(excit_wfm, resp_wfm, 'r', label='Raw')
            axis.plot(excit_wfm, filt_data, 'k', label='Filtered')
            axis.legend(fontsize=14)
            set_tick_font_size(axis, 14)
            axis.set_xlabel('Excitation', fontsize=16)
            axis.set_ylabel('Signal', fontsize=16)
            axis.set_title(plot_title, fontsize=16)
            fig_loops.tight_layout()
        else:
            # N loops:
            raw_pixels = np.reshape(resp_wfm, (-1, len(excit_wfm)))
            filt_pixels = np.reshape(filt_data, (-1, len(excit_wfm)))
            print(raw_pixels.shape, filt_pixels.shape)

            fig_loops, axes_loops = plot_curves(excit_wfm, [raw_pixels, filt_pixels], line_colors=['r', 'k'],
                                                dataset_names=['Raw', 'Filtered'], x_label='Excitation',
                                                y_label='Signal', subtitle_prefix='Col ', num_plots=16,
                                                title=plot_title)

    return filt_data, fig_fft, fig_loops
