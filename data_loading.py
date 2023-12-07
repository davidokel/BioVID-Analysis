import os
import pandas as pd
import numpy as np


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

    print(X.shape)
    print(y.shape)

    return X, y


def augment_data(X, y, fs=256, window_length=4.5, shift_step=0.25, max_shift=1.0):
    """
    Perform time-shifting data augmentation for BioVid Heat Pain Database.

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

    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        # Starting point of the original window
        start_point = max_shift_points

        for shift in range(-max_shift_points, max_shift_points + 1, shift_points):
            # Calculate the starting and ending points of the shifted window
            start = start_point + shift
            end = start + num_points_per_window

            # Extract the shifted window and add it to augmented data
            shifted_window = X[i, start:end, :]
            augmented_X.append(shifted_window)
            augmented_y.append(y[i])

    X = np.array(augmented_X)
    y = np.array(augmented_y)

    print(X.shape)
    print(y.shape)

    return X, y
# %%
