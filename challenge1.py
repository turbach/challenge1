'''coding challenge ... speed up an inefficient function

1. set up a git repo for this challenge ... download or fork 

2. get oriented ... 

  * first run this script as a demo a couple times without change and study the plot
  * read the slow_scan() docstring and comments
  
3. write your quick_scan() function that does the same job as slow_scan()
   and runs as quickly as you can make it go.

3. use timeit or similar to show how much quicker for 

   do_demo == True 
   do_demo == False

4. write a unit test or tests to ensure that slow_scan and quick_scan
   produce the same output on the same input

5. document your quick_scan() for sphinx in numpy style per

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

6. track your progress in git ... branch however you like, commit often.

7. point me to the git hub repo or zip up the directory and email when its done

'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb

def quick_scan(data, win_len, max_excursion):
    '''efficient moving window scan of a data vector for peak-to-peak amplitude excursions

    '''
    raise NotImplementedError

def slow_scan(data, win_len, max_excursion):
    '''inefficienty moving window scan of a data vector for peak-to-peak amplitude excursions
    
    The aim is to separate a data vector into those intervals of data
    that contain extreme peak-to-peak data ranges ("bad") and those
    intervals that do not ("good").

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
    # this is stupid b.c. the same data get tested win_len times
    for i in range(n-win_len):

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

def demo_plot(demo_data, results):
    ''' demo plotting ... not for timing '''

    # make masks and plot good and bad data
    bad_idxs = np.where(results == 1)[0]
    good_idxs = np.where(results == 0)[0]

    bad_data = np.ones_like(demo_data) * np.nan
    bad_data[bad_idxs] = data[bad_idxs]

    good_data = np.ones_like(demo_data) * np.nan
    good_data[good_idxs] = data[good_idxs]

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


if __name__ == '__main__':

    do_demo = 1 # boolean to switch between demo/debugging and simulated data

    # fake an array of digitized EEG as n_samp (rows) x (n_chan) columns
    # where ...
    # 
    #    n_samp : number of A/D samples and 
    #    n_chan : number of EEG data streams

    uV_scale = 25 # used to scale N(0,1) to approximate uV EEG data
    if do_demo:
        # small values for debugging/plotting
        n_samps = int(1e3)   
        n_chans = 1          
    else:
        # large values typical real EEG data
        n_samps = int(1e6) 
        n_chans = 32 

    # random data, smoothed a bit
    b, a = signal.butter(3, 0.5)
    data = uV_scale * np.random.randn(n_samps*n_chans).round(2)
    data = signal.filtfilt(b, a, data)
    data = data.reshape(n_samps, n_chans)

    # length of the scan interval (= "locality" constraint)
    w = 30
    
    # maximum allowed peak-to-peak range within each interval
    m = 70

    # scan each channel for excursions
    print('{0} samples ...'.format(n_samps))
    for c in range(n_chans):
        print('chan {0}/{1}'.format(c,n_chans))
        results = slow_scan(data[:,c], w, m)
    print('done')

    # for demos plot the last column results
    if do_demo:
        demo_plot(data[:,n_chans-1],results)

