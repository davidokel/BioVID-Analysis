from data_loading import load_dataset, create_input_space, split_data_by_candidate, augment_data
from data_processing import preprocess_signals, normalize_data, trim_signals


def loading_pipeline(test_size=5):
    data = load_dataset(signal_names=['ecg', 'gsr'])
    print(f"Number of samples: {len(data)}")
    print(f"Signal Length: {len(data[0]['signals']['ecg'])}")

    data_filtered = preprocess_signals(data, 512, 256)
    train_data, test_data = split_data_by_candidate(data_filtered, test_size)

    X_train, y_train = create_input_space(train_data)
    X_test, y_test = create_input_space(test_data)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    augmented_X, augmented_y = augment_data(X_train, y_train)
    X_test = trim_signals(X_test)

    return augmented_X, augmented_y, X_test, y_test
