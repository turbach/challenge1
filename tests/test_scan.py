import pytest
import numpy as np
from scipy import signal
from timeit import timeit
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


@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
@pytest.mark.parametrize("window", [1, 3, 10, 30])
@pytest.mark.parametrize("max_amplitude", [10, 70])
def test_quick_scan_correctness_random_data(n_samples, window, max_amplitude):
    '''Test output of quick_scan against the output of slow_scan on correctness,
       use randomly generated data.'''
    
    data = generate_data(n_samples, n_channels=1)

    slow_result = slow_scan(data, window, max_amplitude)
    quick_result = quick_scan(data, window, max_amplitude)

    assert (slow_result == quick_result).all()


@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
@pytest.mark.parametrize("window", [1, 3, 10, 30])
@pytest.mark.parametrize("max_amplitude", [10, 70])
def test_quick_scan_speed_random_data(n_samples, window, max_amplitude):
    '''Test execution time of quick scan against execution time of slow_scan,
       use randomly generated data. Succeeds if quick_scan is faster than 
       slow_scan, fails otherwise.'''

    data = generate_data(n_samples, n_channels=1)

    # decorator wizardry to be able to time a function with arguments
    def timeit_decorator(func, data, window, max_amplitude):
        def func_no_args():
            return func(data, window, max_amplitude)
        return func_no_args

    n_runs = 3
    
    slow_scan_no_args = timeit_decorator(slow_scan, data, window, max_amplitude)
    slow_scan_time = timeit(slow_scan_no_args, number=n_runs)

    quick_scan_no_args = timeit_decorator(quick_scan, data, window, max_amplitude)
    quick_scan_time = timeit(quick_scan_no_args, number=n_runs)

    assert quick_scan_time < slow_scan_time


def test_benchmark_slow_scan(benchmark):
    '''Benchmark slow_scan using a pregenerated dataset, then compare against
       precomputed result.'''

    data = np.load('tests/test_data.npy')
    expected_result = np.load('tests/test_result.npy')

    window, max_amplitude = 30, 70

    actual_result = benchmark(slow_scan, data, window, max_amplitude)

    assert (expected_result == actual_result).all()

def test_benchmark_quick_scan(benchmark):
    '''Benchmark quick_scan using a pregenerated dataset, then compare against
       precomputed result.'''

    data = np.load('tests/test_data.npy')
    expected_result = np.load('tests/test_result.npy')

    window, max_amplitude = 30, 70

    actual_result = benchmark(quick_scan, data, window, max_amplitude)

    assert (expected_result == actual_result).all()

