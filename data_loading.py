import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(directory_path="PartB/biosignals_raw/biosignals_raw", signal_names=None):
    if signal_names is None:
        signal_names = ['ecg', 'gsr', 'emg_trapezius', 'emg_corrugator', 'emg_zygomaticus']

    def load_signal(file_path, delimiter='\t'):
        """
        Load raw signal data from a CSV file with custom delimiters.

        Args:
        - file_path (str): The path to the CSV file.
        - delimiter (str, optional): The delimiter used in the CSV file (default is '\t' for tab-delimited).

        Returns:
        - pd.DataFrame: A DataFrame containing the raw signal data.
        """
        try:
            raw_signal_data = pd.read_csv(file_path, delimiter=delimiter)
            return raw_signal_data
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            return None

    data = []

    for candidate in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, candidate)
        # Iterate through all files in the directory
        for filename in os.listdir(folder_path):
            if filename.endswith("_bio.csv"):
                # Extract information from the filename
                parts = filename.split('-')
                age = int(parts[0][-2:])  # Extract age (e.g., 21)
                gender = parts[0][-3]  # Extract gender (e.g., 'w' for woman)

                # Determine the pain level based on the filename
                if "BL" in filename:
                    pain_level = 0  # Assign pain level 0 for baseline
                else:
                    pain_level = int(parts[1][2:])  # Extract pain level (e.g., 2)

                # Read the CSV file and store it in a DataFrame
                file_path = os.path.join(folder_path, filename)
                df = load_signal(file_path)

                # Extract the specified signals
                signals = {signal: df[signal] for signal in signal_names if signal in df}

                # Append the data to the list
                data.append({
                    'age': age,
                    'candidate': candidate,
                    'gender': gender,
                    'pain_level': pain_level,
                    'signals': signals
                })

    return data


def split_data_by_candidate(data, test_size, random_state=42):
    """
    Splits the dataset into training and test sets based on candidate ID.

    :param data: List of dictionaries, each containing 'candidate' and other keys.
    :param test_size: Number of candidates to include in the test set.
    :return: A tuple (train_data, test_data).
    """

    # Group data by candidate
    data_by_candidate = {}
    for record in data:
        candidate = record['candidate']
        if candidate not in data_by_candidate:
            data_by_candidate[candidate] = []
        data_by_candidate[candidate].append(record)

    # Get list of unique candidates
    unique_candidates = list(data_by_candidate.keys())

    # Split candidates into train and test sets
    train_candidates, test_candidates = train_test_split(unique_candidates, test_size=test_size,
                                                         random_state=random_state)

    # Split data into train and test based on candidates
    train_data = [item for candidate in train_candidates for item in data_by_candidate[candidate]]
    test_data = [item for candidate in test_candidates for item in data_by_candidate[candidate]]

    return train_data, test_data


def create_input_space(data):
    # Preallocate lists for X and y, assuming we know the length of data
    num_entries = len(data)
    X = [None] * num_entries  # Preallocate a list of correct size
    y = np.empty(num_entries, dtype=int)  # Preallocate NumPy array for y

    for idx, entry in enumerate(data):
        # Extract and reshape each signal, then concatenate
        signal_data = [signal.reshape(-1, 1) for signal in entry['signals'].values()]
        X[idx] = np.hstack(signal_data)
        y[idx] = entry['pain_level']

    # Convert list of arrays to a NumPy array of objects
    X = np.array(X, dtype=object)

    print(f"X-Shape: {X.shape}")
    print(f"y-Shape: {y.shape}")

    return X, y


def augment_data(X, y, fs=256, window_length=4.5, shift_step=0.25, max_shift=1.0):
    """
    Perform time-shifting data augmentation while maintaining class balance.

    Args:
    - X (np.array): Original data array of shape (num_samples, num_timepoints, num_signals).
    - y (np.array): Labels array.
    - fs (int): Sampling frequency.
    - window_length (float): Length of the window in seconds.
    - shift_step (float): Step for shifting the window in seconds.
    - max_shift (float): Maximum shift in seconds in each direction.

    Returns:
    - np.array: Augmented data array.
    - np.array: Augmented labels array.
    """
    num_points_per_window = int(window_length * fs)
    shift_points = int(shift_step * fs)
    max_shift_points = int(max_shift * fs)

    # Calculate the total number of augmented samples per original sample
    num_shifts_per_sample = (2 * max_shift_points // shift_points) + 1

    # Pre-allocate the arrays
    total_samples = len(X) * num_shifts_per_sample
    augmented_X = np.zeros((total_samples, num_points_per_window, X.shape[2]))
    augmented_y = np.zeros(total_samples, dtype=int)

    sample_index = 0
    for i in range(len(X)):
        for shift in range(-max_shift_points, max_shift_points + 1, shift_points):
            start = max_shift_points + shift
            end = start + num_points_per_window

            if start >= 0 and end <= X.shape[1]:
                shifted_window = X[i, start:end, :]
                augmented_X[sample_index] = shifted_window
                augmented_y[sample_index] = y[i]
                sample_index += 1

    # Trim arrays in case some windows did not fit
    augmented_X = augmented_X[:sample_index]
    augmented_y = augmented_y[:sample_index]

    print(f"Augmented X.shape: {augmented_X.shape}")
    print(f"Augmented y.shape: {augmented_y.shape}")

    return augmented_X, augmented_y
# %%
