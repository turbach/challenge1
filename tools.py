import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb

def quick_scan(data, win_len, max_excursion):
    '''efficient moving window scan of a data vector for peak-to-peak amplitude excursions

    '''
    raise NotImplementedError

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

            result[(i+bad_start):(i+bad_stop)] = 1

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


