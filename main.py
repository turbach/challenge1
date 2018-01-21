'''coding challenge ... speed up an inefficient function

1. set up a git repo for this challenge ... download or fork 

2. get oriented ... 

  * first run this script as a demo a couple times without change and study the plot
  * read the slow_scan() docstring and comments
  
3. fill in the quick_scan() function stub so that it does the same job 
   as slow_scan() and runs as quickly as you can make it go.

3. use timeit or similar to show how much quicker for 

   do_demo == True 
   do_demo == False

4. write a unit test or tests to ensure that slow_scan and quick_scan
   produce the same output on the same input

5. document your quick_scan() for sphinx in numpy style per

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
    http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

6. track your progress in git ... branch however you like, commit often.

7. point me to the git repo or zip up the directory and email when its done

'''
import numpy as np
from scipy import signal
from tools import quick_scan, slow_scan, demo_plot

do_demo = 1 # boolean to switch between demo/debugging and simulated data

# fake an array of digitized EEG as n_samp (rows) x (n_chan) columns
# where ...
# 
#    n_samp : number of A/D samples and 
#    n_chan : number of EEG data streams

uV_scale = 25 # cosmetic only, scales N(0,1) to approximate uV EEG data
if do_demo:
    # small values for debugging/plotting
    n_samps = int(1e3)   
    n_chans = 1          
else:
    # larger values more typical of real EEG data
    n_samps = int(1e6) 
    n_chans = 32 

# fetch random data, smoothed somewhat
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
    demo_plot(data[:,n_chans-1],results, uV_scale)

