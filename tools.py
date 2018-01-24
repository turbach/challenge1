import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import bottleneck as bn


def quick_scan(data, win_len, max_excursion):
    """Scan data for amplitude excursions within windows of fixed size.

    Within each window, the edges of the excursion subinterval are detected,
    data within each subinterval is marked as 'bad'.

    Parameters
    ----------
    data : np.array, shape=(n,)
        1-D data array of length n
    win_len : positive int
        size of the rolling window to be tested for excursions
    max_excursion : positive int
        maximum peak-to-peak range allowed

    Returns
    -------
    result : np.array, shape=(n,), dtype=bool
        array of bools, same length as the input data; True at the indices of
        the bad stretches, False otherwise.


    """

    # make sure the data is a flat array
    data = data.reshape(-1)

    # prepare result array
    n = len(data)
    result = np.full(n, False)

    # detect amplitude excursions
    bad_windows_mask = get_bad_windows_mask(data, win_len, max_excursion)

    # return if no excursions
    if not bad_windows_mask.any():
        return result

    # subset windows where excursions occur
    bad_windows = strided_view(data, win_len)[bad_windows_mask]

    # get locations of bad data
    bad_indices = get_bad_indices(n, win_len, bad_windows, bad_windows_mask)

    # set flags at 'bad' data locations
    result[bad_indices] = True

    return result


def get_bad_windows_mask(data, win_len, max_excursion):
    """Get boolean mask corresponding to windows with 'bad' data.

    Parameters
    ----------
    data : 1D array, shape=(n,)
        1-D data array of length n
    win_len : positive int
        size of the rolling window to be tested for excursions
    max_excursion : positive int
        maximum peak-to-peak range allowed

    Returns
    -------

    bad_windows_mask : np.array, shape=(n-win_len+1,), dtype=bool
        array of bools, same length as the number of rolling windows; True at
        the indices of bad windows, False otherwise

    """

    # calculate amplitude within rolling windows
    maxima = bn.move_max(data, window=win_len)
    minima = bn.move_min(data, window=win_len)
    amplitude = (maxima - minima)[win_len-1:]

    # detect amplitude excursions
    bad_windows_mask = amplitude > max_excursion

    return bad_windows_mask


def get_bad_indices(n, win_len, bad_windows, bad_windows_mask):
    """Create an array containing indices of bad data.

    Parameters
    ----------
    n : positive int
        length of data
    win_len : positive int
        size of the rolling window
    bad_windows : np.array, shape=(n_bad_windows, win_len)
        2D numeric array of views of windows where bad data was found
    bad_windows_mask : np.array, shape=(n-win_len+1,), dtype=bool
        array of bools, same length as the number of rolling windows; True at
        the indices of bad windows, False otherwise

    Returns
    -------
    bad_indices : np.array, shape=(n,), dtype=int
        array of ints indicating the locations of bad data

    """

    # detect rightmost extrema within flagged windows
    rmin = win_len - 1 - bn.move_argmin(bad_windows, win_len)[:, win_len-1]
    rmax = win_len - 1 - bn.move_argmax(bad_windows, win_len)[:, win_len-1]

    # detect leftmost extrema within flagged windows
    flipped_bad_windows = np.flip(bad_windows, axis=1)
    lmin = bn.move_argmin(flipped_bad_windows, win_len)[:, win_len-1]
    lmax = bn.move_argmax(flipped_bad_windows, win_len)[:, win_len-1]

    # calculate relative (within windows) indices of subintervals of 'bad' data
    rel_bad_starts = np.minimum(lmin, lmax).astype(int)
    rel_bad_stops = np.maximum(rmin, rmax).astype(int)

    # calculate absolute (within channel) indices of subintervals of 'bad' data
    abs_offsets = np.arange(n - win_len + 1)
    abs_bad_starts = abs_offsets[bad_windows_mask] + rel_bad_starts
    abs_bad_stops = abs_offsets[bad_windows_mask] + rel_bad_stops + 1

    # construct list of paired [start, stop] indices, detect unique
    slices = np.array((abs_bad_starts, abs_bad_stops)).T
    unique_slices = np.unique(slices, axis=0)

    # construct array index from paired indices
    bad_interval_indices = np.array([np.arange(*s) for s in unique_slices])
    bad_indices = np.unique(np.concatenate(bad_interval_indices))

    return bad_indices


def strided_view(data, win_len):
    """Create an array of rolling window views (no copying).

    For more information, see the docs of numpy.lib.stride_tricks.as_strided.

    Parameters
    ----------
    data : np.array, shape=(n,)
        1-D data array of length n
    win_len : positive int
        size of the rolling window to be tested for excursions

    Returns
    -------
    strided_view : np.array, shape=(n-win_len+1, win_len)
        a view of the data 'through' rolling windows of width win_len
    """

    shape = (len(data) - win_len + 1, win_len)
    stride, = data.strides
    strides = (stride, stride)
    strided_view = as_strided(data, shape=shape, strides=strides)
    return strided_view


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


