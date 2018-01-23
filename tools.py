import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal
import matplotlib.pyplot as plt
import bottleneck as bn


def quick_scan(channel, window, max_amplitude):

    # make sure the data is a flat array
    channel = channel.reshape(-1)
    n = len(channel)

    # prepare result array
    result = np.zeros(n)
    
    # detect amplitude excursions
    bad_windows_mask = get_bad_windows_mask(channel, window, max_amplitude)

    # return if no excursions
    if bad_windows_mask.any() == False:
        return result

    # subset windows where excursions occur
    flagged_windows = strided_view(channel, window)[bad_windows_mask]

    # get relative bounds of bad intervals
    rel_bad_starts, rel_bad_stops = get_rel_bad_bounds(flagged_windows, window)

    # calculate absolute (within channel) indices of subintervals of 'bad' data
    absolute_offsets = np.arange(n - window + 1)
    abs_bad_starts = absolute_offsets[bad_windows_mask] + rel_bad_starts
    abs_bad_stops  = absolute_offsets[bad_windows_mask] + rel_bad_stops + 1

    # get locations of bad data
    index_array = get_bad_indices(abs_bad_starts, abs_bad_stops)

    # set flags at 'bad' data locations
    result[index_array] = 1
    
    return result


def get_bad_windows_mask(channel, window, max_amplitude):

    # calculate amplitude within rolling windows
    maxima = bn.move_max(channel, window=window)
    minima = bn.move_min(channel, window=window)
    amplitude = (maxima - minima)[window-1:]

    # detect amplitude excursions
    bad_windows_mask = amplitude > max_amplitude

    return bad_windows_mask


def get_rel_bad_bounds(flagged_windows, window):

    # detect rightmost extrema within flagged windows
    rightmost_min = window - 1 - bn.move_argmin(flagged_windows, window)[:, window-1]
    rightmost_max = window - 1 - bn.move_argmax(flagged_windows, window)[:, window-1]

    # detect leftmost extrema within flagged windows
    flipped_flagged_windows = np.flip(flagged_windows, axis=1)
    leftmost_min = bn.move_argmin(flipped_flagged_windows, window)[:, window-1]
    leftmost_max = bn.move_argmax(flipped_flagged_windows, window)[:, window-1]

    # calculate relative (within windows) indices of subintervals of 'bad' data
    rel_bad_starts = np.minimum(leftmost_min, leftmost_max)
    rel_bad_stops  = np.maximum(rightmost_min, rightmost_max)

    return rel_bad_starts, rel_bad_stops


def get_bad_indices(abs_bad_starts, abs_bad_stops):

    # construct list of paired [start, stop] indices, detect unique
    slices = np.array((abs_bad_starts, abs_bad_stops)).astype(int).T
    unique_slices = np.unique(slices, axis=0)

    # construct array index from paired indices
    bad_interval_indices = np.array([np.arange(*s) for s in unique_slices])
    index_array = np.unique(np.concatenate(bad_interval_indices))

    return index_array


def strided_view(data, win_len):
    
    shape = (len(data) - win_len + 1, win_len)
    stride, = data.strides
    strides = (stride, stride)
    return as_strided(data, shape=shape, strides=strides)


def slow_scan(data, win_len, max_excursion):
    '''inefficient moving window scan of a data vector for peak-to-peak amplitude excursions
    
    The aim is to tag intervals that contain extreme peak-to-peak 
    data ranges ("bad") vs. intervals that do not ("good").

    Parameters
    ----------
    data : np.array, shape=(n,), dtype=float 
        1-D data vector of length n
    win_len : uint
        length of the data sub-intervals to test for excursions
    max_excursion : uint
        maximum peak-to-peak range allowed

    Returns
    -------
    result : np.array, shape=(n,), dtype=bool
       vector of bools, same length as the input data. True at the indices of
       the bad stretchs, False otherwise.

    '''

    n = len(data)
    result = np.zeros(n)

    # scan the data for excursions in sliding windows, stepping by 1
    # dopey ... data are re-tested needlessly and with slow lookups.
    for i in range(n-win_len+1):

        this_interval = data[i:i+win_len] # a slice of data to check
        this_max = this_interval.max()
        this_min = this_interval.min()

        if np.abs(this_max - this_min) > max_excursion:

            # min, max not necessarily unique in the interval
            min_idxs = np.where(this_interval == this_min)[0]
            max_idxs = np.where(this_interval == this_max)[0]
            bad_idxs = np.append(min_idxs, max_idxs)

            # min, max values may occur in any order ...
            # flag everything in between as bad
            bad_start = bad_idxs.min()
            bad_stop = bad_idxs.max()

            result[(i+bad_start):(i+bad_stop+1)] = 1

    return result

def demo_plot(demo_data, results, uV_scale):
    ''' demo plotting ... not for timing '''

    # make masks and plot good and bad data
    bad_idxs = np.where(results == 1)[0]
    good_idxs = np.where(results == 0)[0]

    bad_data = np.ones_like(demo_data) * np.nan
    bad_data[bad_idxs] = demo_data[bad_idxs]

    good_data = np.ones_like(demo_data) * np.nan
    good_data[good_idxs] = demo_data[good_idxs]

    plt.close('all')
    plt.figure()

    plt.subplot(311)
    plt.title('demo data')
    plt.plot(demo_data, 'b')

    plt.subplot(312)
    plt.title('bad intervals contain extreme peak-to-peak ranges')
    plt.plot(demo_data, 'b', alpha=.2)
    plt.plot(bad_data, 'r')
    plt.plot(results * uV_scale, 'k')

    plt.subplot(313)
    plt.title('good intervals had better not')
    plt.plot(demo_data, 'b', alpha=.2)
    plt.plot(good_data, 'm')
    plt.plot(results * uV_scale, 'k')

    plt.show()


