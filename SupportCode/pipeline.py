from sklearn.model_selection import train_test_split

from data_loading import load_dataset, create_input_space, split_data_by_candidate, augment_data
from data_processing import preprocess_signals, normalize_data, trim_signals
from sklearn.utils import shuffle


def loading_pipeline(test_size=5, augment=True, holdout=True, downsample=256):
    data = load_dataset(signal_names=['ecg', 'gsr'])
    print(f"Number of samples: {len(data)}")
    print(f"Signal Length: {len(data[0]['signals']['ecg'])}")

    data_filtered = preprocess_signals(data, 512, downsample)

    if not holdout:
        # Split the data without considering candidate ID
        all_data = create_input_space(data_filtered)
        X, y = all_data
        X = normalize_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/len(data), random_state=42)
    else:
        # Split the data by candidate ID
        train_data, test_data = split_data_by_candidate(data_filtered, test_size)
        X_train, y_train = create_input_space(train_data)
        X_test, y_test = create_input_space(test_data)
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    if augment:
        X_train, y_train = augment_data(X_train, y_train, downsample)
        X_test = trim_signals(X_test, downsample)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    return X_train, y_train, X_test, y_test
