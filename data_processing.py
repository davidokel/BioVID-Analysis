from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import butter, filtfilt, resample


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
