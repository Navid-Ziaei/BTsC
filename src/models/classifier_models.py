
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

    def fit(self, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set the model in training mode
        self.train()

        for epoch in range(num_epochs):
            train_loss = 0.0
            train_correct = 0
            total_train_samples = 0

            for inputs, targets in train_loader:
                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)

                # Compute the loss
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update training statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                total_train_samples += inputs.size(0)

            # Compute training accuracy and average loss
            train_accuracy = train_correct / total_train_samples
            average_train_loss = train_loss / total_train_samples

            # Evaluate on the validation set
            with torch.no_grad():
                self.eval()
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / len(y_val)

            # Print the training and validation metrics for monitoring progress
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: "
                f"{train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Return the trained model
        return self


class BaysianModel:
    def __init__(self):
        self.label_list = None
        self.num_categories = None
        self.num_feature = None

        # Trainable variables
        self.mu = None
        self.cov = None

    def fit(self, x_train, y_train):
        self.label_list = list(np.unique(y_train))
        # get input size
        self.num_categories = len(self.label_list)
        self.num_feature = x_train.shape[-1]

        # initiate trainable variables
        self.mu = np.zeros((self.num_categories, 1, self.num_feature))
        self.cov = np.zeros((self.num_categories, self.num_feature, self.num_feature))
        for label_idx, label in enumerate(self.label_list):
            idx = np.where(y_train == label)[0]
            self.mu[label_idx] = np.mean(x_train[idx], axis=0, keepdims=True)
            self.cov[label_idx] = np.matmul(np.transpose(x_train[idx] - self.mu[label_idx]),
                                            x_train[idx] - self.mu[label_idx]) / len(idx)

    def predict_proba(self, x_test):
        if self.mu is None:
            raise ValueError("Model is not trained")
        predicted_prob = []
        for i in range(self.num_categories):
            predicted_prob.append(
                multivariate_normal.pdf(x_test, mean=np.squeeze(self.mu[i]), cov=self.cov[i], allow_singular=True))
        predicted_prob = np.stack(predicted_prob, axis=-1)
        predicted_prob = predicted_prob / (np.sum(predicted_prob, axis=-1, keepdims=True) + 1e-8)

        return predicted_prob

    def predict(self, x_test):
        y_prob = self.predict_proba(x_test)
        y_label = np.argmax(y_prob, axis=-1)
        return y_label

    def evaluate(self, x_test, y_test, metric='acc'):
        y_hat = self.predict(x_test)
        result = {}
        if metric == 'acc':
            result['acc'] = accuracy_score(y_true=y_test, y_pred=y_hat)
        else:
            raise ValueError("Not implemented")

        return result
