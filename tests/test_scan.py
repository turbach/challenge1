import pytest
import numpy as np
from scipy import signal
from ..tools import slow_scan, quick_scan

def generate_data(n_samples, n_channels):

    # scale to approximate real EEG
    uV_scale = 25

    # get Butterfield filter coefficients
    b, a = signal.butter(3, 0.5)

    # sample, apply filter, and reshape 
    data = uV_scale * np.random.randn(n_samples*n_channels).round(2) 
    data = signal.filtfilt(b, a, data)
    data = data.reshape(n_samples, n_channels)

    return data

def test_quick_scan_random_data():
    '''Test output of quick_scan against the output of slow_scan using randomly
       generated data.'''

    data = generate_data(1000, 1)
    window = 30
    maxamp = 70 # maximum peak-to-peak amplitude

    assert slow_scan(data, window, maxamp) == quick_scan(data, window, maxamp)

