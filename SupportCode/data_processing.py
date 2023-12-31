from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import butter, filtfilt, resample


def trim_signals(X, fs=256, window_length=4.5):
    """
    Trim or extend signals in the dataset to a specified window length.

    Args:
    - X (np.array): Original data array of shape (num_samples, num_timepoints, num_signals).
    - fs (int): Sampling frequency.
    - window_length (float): Desired length of the window in seconds.

    Returns:
    - np.array: Trimmed or extended data array.
    """
    num_points_per_window = int(window_length * fs)
    num_samples, num_timepoints, num_signals = X.shape

    # Pre-allocate the array
    trimmed_X = np.zeros((num_samples, num_points_per_window, num_signals))

    for i in range(num_samples):
        if num_timepoints > num_points_per_window:
            # Trim the signal if it's longer than the desired window
            trimmed_X[i] = X[i, :num_points_per_window, :]
        else:
            # Extend the signal with zeros if it's shorter than the desired window
            trimmed_X[i, :num_timepoints] = X[i]

    print(f"Trimmed X.shape: {trimmed_X.shape}")
    return trimmed_X


def normalize_data(X, local=True):
    """
    Normalize each signal in the data either locally or globally.

    Args:
    - X (np.array): The raw data, expected shape (num_samples, num_timepoints, num_signals).
    - local (bool): 'True' for sample-wise normalization, 'False' for global normalization.

    Returns:
    - np.array: Normalized data.
    """
    X_normalized = []
    num_signals = X.shape[2]

    if not local:
        # Global normalization: Fit scalers on all data for each signal
        global_scalers = [MinMaxScaler().fit(X[:, :, i].reshape(-1, 1)) for i in range(num_signals)]

    for sample in X:
        normalized_signals = []

        for i in range(num_signals):
            signal_data = sample[:, i].reshape(-1, 1)

            if local:
                # Local normalization: Scale data sample-wise
                scaler = MinMaxScaler().fit(signal_data)
                normalized_signal = scaler.transform(signal_data)
            else:
                # Global normalization: Scale data with global scalers
                normalized_signal = global_scalers[i].transform(signal_data)

            normalized_signals.append(normalized_signal)

        # Concatenate all normalized signals horizontally for each sample
        X_normalized.append(np.hstack(normalized_signals))

    return np.array(X_normalized)


def butter_filter(data, lowcut, highcut, fs, order=3, btype='band'):
    nyq = 0.5 * fs

    # Define the frequency cutoffs for the filter
    if btype == 'low':
        norm_cutoff = highcut / nyq
    elif btype == 'high':
        norm_cutoff = lowcut / nyq
    else:  # 'band' and 'stop' types require a list of [low, high] cutoffs
        norm_cutoff = [lowcut / nyq, highcut / nyq]

    b, a = butter(order, norm_cutoff, btype=btype)
    y = filtfilt(b, a, data)
    return y


def piecewise_detrend(data, poly_order=5):
    # Creating a time vector
    t = np.arange(data.shape[0])
    # Fitting the polynomial
    p = np.polyfit(t, data, poly_order)
    # Evaluating the polynomial
    trend = np.polyval(p, t)
    # Detrending
    detrended_data = data - trend
    return detrended_data


def preprocess_signals(data, fs, down_sample=None):
    """
    Preprocess the physiological signals as per the specified steps.

    Args:
    - data (list of dicts): List containing the data and signals.
    - fs (int): Sampling frequency.

    Returns:
    - list of dicts: Preprocessed data.
    """
    preprocessed_data = []

    for entry in data:
        processed_signals = {}

        for signal_key, signal in entry['signals'].items():
            if down_sample is not None:
                # Calculate the number of points to resample
                num_points = int(len(signal) * down_sample / fs)
                signal = resample(signal, num_points)

            if signal_key == 'ecg':
                filtered = butter_filter(signal, 0.1, 250, fs, order=3, btype='band')
                detrended = piecewise_detrend(filtered)
                processed_signals['ecg'] = detrended

            elif signal_key == 'gsr':
                filtered = butter_filter(signal, None, 0.2, fs, order=3, btype='low')
                processed_signals['gsr'] = filtered

            elif 'emg' in signal_key:  # For EMG signals
                filtered = butter_filter(signal, 20, 250, fs, order=4, btype='band')
                processed_signals[signal_key] = filtered

        preprocessed_entry = {
            'age': entry['age'],
            'candidate': entry['candidate'],
            'gender': entry['gender'],
            'pain_level': entry['pain_level'],
            'signals': processed_signals
        }
        preprocessed_data.append(preprocessed_entry)

    return preprocessed_data


def plot_signal_samples(data, data_filtered, sample_indices, fs=256, fs_filtered=256):
    """
    Plot original, filtered, and difference for all signal types in given samples.

    Args:
    - data (list of dicts): Original signals.
    - data_filtered (list of dicts): Filtered signals.
    - sample_indices (list): Indices of the samples to plot.
    - fs (int): Sampling frequency for original signals for correct time axis plotting.
    - fs_filtered (int): Sampling frequency for filtered signals for correct time axis plotting.
    """
    for idx in sample_indices:
        signals = data[idx]['signals'].keys()
        num_signals = len(signals)

        fig, axs = plt.subplots(num_signals, 3, figsize=(15, 4 * num_signals), constrained_layout=True)
        fig.suptitle(f'Sample {idx}', fontsize=16)

        for i, key in enumerate(signals):
            # Original signal time axis
            time_original = np.arange(len(data[idx]['signals'][key])) / fs
            # Filtered signal time axis
            time_filtered = np.arange(len(data_filtered[idx]['signals'][key])) / fs_filtered

            axs[i, 0].plot(time_original, data[idx]['signals'][key], label='Original')
            axs[i, 0].set_title(f'{key.capitalize()} Signal - Original')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Amplitude')

            axs[i, 1].plot(time_filtered, data_filtered[idx]['signals'][key], label='Filtered')
            axs[i, 1].set_title(f'{key.capitalize()} Signal - Filtered')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('Amplitude')

            # Resample the original signal to match the sampling rate of the filtered signal
            resampled_original_signal = resample(data[idx]['signals'][key], len(data_filtered[idx]['signals'][key]))
            time_diff = np.arange(len(resampled_original_signal)) / fs_filtered
            diff_signal = resampled_original_signal - data_filtered[idx]['signals'][key]

            axs[i, 2].plot(time_diff, diff_signal, label='Difference')
            axs[i, 2].set_title(f'{key.capitalize()} Difference (Original - Filtered)')
            axs[i, 2].set_xlabel('Time (s)')
            axs[i, 2].set_ylabel('Amplitude')

        plt.show()
