import copy

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, roc_curve
)
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class Model:
    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def predict_proba(self, X_test):
        # Default implementation assumes predict method returns probabilities
        return self.predict(X_test)


def _create_dataloader(X, y, batch_size=64, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class PyTorchModel(Model):
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self, X_train, y_train, num_epochs=25, validation_split=0.1, patience=5):
        self.model.to(self.device)

        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=validation_split
        )

        train_loader = _create_dataloader(X_train_split, y_train_split)
        val_loader = _create_dataloader(X_val_split, y_val_split, shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation loop
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping triggered')
                    break

    def predict(self, X_test):
        self.model.to(self.device)
        self.model.eval()
        test_loader = _create_dataloader(X_test, np.zeros(len(X_test)), batch_size=128, shuffle=False)
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted.cpu().numpy())
        return np.concatenate(predictions)

    def predict_proba(self, X_test):
        self.model.to(self.device)
        self.model.eval()
        test_loader = _create_dataloader(X_test, np.zeros(len(X_test)), batch_size=128, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
        return np.concatenate(probabilities)


def filter_for_binary_task(X, y, label_of_interest):
    # Convert to binary task - one class vs all others
    y_binary = (y != label_of_interest).astype(int)
    return X, y_binary


def filter_and_convert_labels(X, y, labels_to_keep):
    indices_to_keep = np.isin(y, labels_to_keep)
    X_filtered = X[indices_to_keep]
    y_filtered = y[indices_to_keep]
    y_filtered = (y_filtered == max(labels_to_keep)).astype(int)  # Convert the highest label to 1
    return X_filtered, y_filtered


def pretrain_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(base_model, X_train_pre, y_train_pre, X_train, y_train, X_test, y_test):
    tasks = {
        "T0 vs T1 vs T2 vs T3 vs T4": [0, 1, 2, 3, 4],
        "T0 vs (T1, T2, T3, T4)": 0,  # Treat T0 vs all others as a binary classification
        "T0 vs T1": [0, 1],
        "T0 vs T2": [0, 2],
        "T0 vs T3": [0, 3],
        "T0 vs T4": [0, 4]
    }

    results = {}

    # Pretraining on the entire dataset
    pretrain_model(copy.deepcopy(base_model), X_train_pre, y_train_pre)

    for task_name, labels in tasks.items():
        # Clone the pretrained model for fine-tuning
        model = copy.deepcopy(base_model)

        if isinstance(labels, list):
            if len(labels) > 2:  # Multiclass classification
                X_train_fine, y_train_fine = X_train, y_train
                X_test_fine, y_test_fine = X_test, y_test
            else:  # Binary classification
                X_train_fine, y_train_fine = filter_and_convert_labels(X_train, y_train, labels)
                X_test_fine, y_test_fine = filter_and_convert_labels(X_test, y_test, labels)
        else:  # Binary task T0 vs (T1, T2, T3, T4)
            X_train_fine, y_train_fine = filter_for_binary_task(X_train, y_train, labels)
            X_test_fine, y_test_fine = filter_for_binary_task(X_test, y_test, labels)

            # Compute class weights for the binary classification task
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_fine),
                y=y_train_fine
            )
            # Convert class weights to a tensor
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(model.device)

            # Update your model's criterion to include the class weights
            # This assumes your criterion can accept class weights (like torch.nn.CrossEntropyLoss)
            model.criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        # Fine-tuning on the specific task
        model.fit(X_train_fine, y_train_fine)

        # Evaluation
        predictions = model.predict(X_test_fine)
        accuracy = accuracy_score(y_test_fine, predictions)
        precision = precision_score(y_test_fine, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test_fine, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test_fine, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test_fine, predictions)

        results[task_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": cm
        }

        # ROC for binary classification
        if len(set(y_test_fine)) == 2:
            roc_auc = roc_auc_score(y_test_fine, predictions)
            fpr, tpr, _ = roc_curve(y_test_fine, model.predict_proba(X_test_fine)[:, 1])
            results[task_name]["ROC-AUC"] = roc_auc
            results[task_name]["ROC Curve"] = (fpr, tpr)

    return results


def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # MSE, RMSE, MAE, R2
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE: {mse}\nRMSE: {rmse}\nMAE: {mae}\nR²: {r2}")


def display_results(results):
    # Create a DataFrame for the table
    table_data = {}
    for task, metrics in results.items():
        table_data[task] = {metric: value for metric, value in metrics.items()
                            if metric not in ["Confusion Matrix", "ROC Curve"]}

    df = pd.DataFrame(table_data).T
    print("Performance Metrics Table:")
    display(df)

    for task, metrics in results.items():
        # Confusion Matrices
        if "Confusion Matrix" in metrics:
            plt.figure(figsize=(6, 5))
            sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {task}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

            # ROC Curve for binary classification tasks
        if "ROC Curve" in metrics:
            plt.figure(figsize=(6, 5))
            fpr, tpr = metrics["ROC Curve"]
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["ROC-AUC"]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {task}')
            plt.legend(loc="lower right")
            plt.show()
