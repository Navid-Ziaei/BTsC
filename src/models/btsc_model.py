# user defined packages

from .classifier_models import LSTMClassifier, BaysianModel
# sklearn packages
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import pickle
import seaborn
import matplotlib
from tqdm import tqdm
import numpy as np
import pandas as pd

# Visualize throgh time
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 80}

matplotlib.rc('font', **font)
fontsize = 20

seaborn.set(style="white", color_codes=True)


class BTsCModel:
    """
    1 - fit(self, x_train, y_train, x_test=None, y_test=None, n_splits_outer=5, n_splits_inner=5)
        Trains the model using k-fold cross-validation (when x_test is None) or a single fold. It returns a dictionary
        containing the history of accuracy values during training.
    2 - predict(self, x_test, num_selected_channels=None)
        for the given test data. It returns the predicted labels and the outputs of each selected channel.
    3 - evaluate(self, x_test, y_test, num_selected_channels=None)
        performance of the classifier on the given test data. It returns the accuracy score.
    4 - build_clf_model(self)
        Builds and returns a classifier model based on the specified classifier name.
    5 - save_model(self, save_path)
        save the trained model
    6 - load_model_and_evaluate(self, model_path)
        Builds and returns a classifier model based on the specified classifier name.
    7 - get_best_features_per_channel(self, x_train, y_train, inner_cv, outer_fold)
        Uses K-fold cross-validation to choose the best features for each channel and their performance.
    8 - get_outputs_in_nested_kfold(self, x_train, y_train, n_splits_outer, n_splits_inner)
        Performs nested K-fold cross-validation to obtain outputs and predicted labels for each channel.
        It creates the outer K-fold validation
        in each fold perform get_best_features_per_channel
        For each one it trains the classifier on the outer fold and save the results and all output predictions!
    9 - combine_channels(self, x_train, y_train, n_splits_outer, n_splits_inner, verbose=True)
        get all results in outer k-fold classification
        combine the channels that results in the best performance in average k-outer folds use greedy search
    10- get_combined_results(self, prediction)
        It use maxvote or pol to combine the results

    """

    def __init__(self, settings):
        """
        Constructor for BTsCModel class

        Args:
            classifier_name (str): Name of the classifier to use, defaults to 'bayesian_model'
        """
        # Initialize instance variables
        self.report_model_error = dict()
        self.report_optimum_time = dict()
        self.error_analysis_mode = True
        self.optimum_number_of_channels = None
        self.trial_idx_train = None
        self.num_channels = None
        self.name = 'BTsC'
        self.classifier_name = settings.classifier_name
        self.classifiers = []
        self.selected_channels = []
        self.selected_classifier = []
        self.selected_length = []
        self.best_time_per_channel = dict()

        self.combination_mode = settings.combination_mode
        self.error_analysis = None

    def fit(self, x_train, y_train, trial_idx_train=None, x_test=None, y_test=None, trial_idx_test=None,
            n_splits_outer=5, n_splits_inner=5, channel_names=None):
        """
        Fit the BTsC model on the given data

        Args:
            x_train (ndarray): Training data of shape (n_samples, n_channels, n_timepoints)
            y_train (ndarray): Training labels of shape (n_samples,)
            trial_idx_train (pandas Dataframe): to recorde the trial index
            x_test (ndarray): Test data of shape (n_samples, n_channels, n_timepoints), defaults to None
            y_test (ndarray): Test labels of shape (n_samples,), defaults to None
            trial_idx_test (pandas Dataframe): to recorde the trial index
            n_splits_outer (int): Number of outer cross-validation folds, defaults to 5
            n_splits_inner (int): Number of inner cross-validation folds, defaults to 5

        Returns:
            dict: A dictionary containing the history of accuracy values during training.

        Example:
            >>> model = BTsC()
            >>> x_train = np.random.rand(100, 3, 500)  # 100 samples, 3 channels, 500 timepoints
            >>> y_train = np.random.randint(100)  # Binary labels for 100 samples
            >>> x_test = np.random.rand(50, 3, 500)  # 50 test samples
            >>> y_test = np.random.randint(50)  # Binary labels for 50 test samples
            >>> history = model.fit(x_train, y_train, x_test, y_test, n_splits_outer=10, n_splits_inner=5)

        """
        if trial_idx_train is None:
            trial_idx_train = pd.DataFrame({'Trial number': np.arange(0, x_train.shape[0])})
        self.trial_idx_train = trial_idx_train
        print("Training model...")

        # Set the number of channels in the instance variable
        self.num_channels = x_train.shape[1]

        # Create an inner cross-validation splitter
        inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        # self.error_analysis = pd.DataFrame(0, index=channel_names, columns=list(trial_idx_train.astype(int)))
        # If test data is provided, perform feature selection and training
        selected_channels, selected_channels_mean_accuracy, selected_channels_std_accuracy, \
            best_features_per_channel_all_folds = self.combine_channels(x_train=x_train,
                                                                        y_train=y_train,
                                                                        trial_idx_train=trial_idx_train,
                                                                        outer_cv=outer_cv,
                                                                        inner_cv=inner_cv,
                                                                        verbose=False)

        # Perform feature selection for the training data
        best_features_per_channel_train, single_channel_all_accuracy = self.get_best_features_per_channel(
            x_train=x_train,
            y_train=y_train,
            trial_idx_train=trial_idx_train,
            inner_cv=inner_cv,
            outer_fold='test')



        # Clear the selected length and classifier lists
        self.selected_length = []
        self.selected_classifier = []
        self.selected_channels = selected_channels

        # Train classifiers for each selected channel
        selected_classifier_test_accuracy, selected_classifier_train_accuracy, selected_classifier_error_analysis = [], [], []

        for ch in selected_channels:
            clf = self.build_clf_model()
            clf.fit(x_train[:, ch, :best_features_per_channel_train[ch]], np.squeeze(y_train))
            self.selected_length.append(best_features_per_channel_train[ch])
            self.selected_classifier.append(clf)
            y_test_pred = clf.predict(x_test[:, ch, :best_features_per_channel_train[ch]])
            y_train_pred = clf.predict(x_train[:, ch, :best_features_per_channel_train[ch]])

            if self.classifier_name.lower() == 'svc':
                p = np.array(clf.decision_function(
                    x_test[:, ch, :best_features_per_channel_train[ch]]))  # decision is a voting function
                prob = (p - np.min(p)) / (np.max(p) - np.min(p))  # softmax after the voting
            else:
                prob = clf.predict_proba(x_test[:, ch, :best_features_per_channel_train[ch]])[:, 1]
            trial_error = np.abs(y_test - prob)

            selected_classifier_test_accuracy.append(accuracy_score(y_true=y_test,
                                                                    y_pred=y_test_pred))
            selected_classifier_train_accuracy.append(accuracy_score(y_true=y_train,
                                                                     y_pred=y_train_pred))
            selected_classifier_error_analysis.append(trial_error)

        selected_channel_names = [channel_names[ch] for ch in self.selected_channels]
        #error_analysis = pd.DataFrame(np.stack(selected_classifier_error_analysis, axis=0),
        #                              index=selected_channel_names, columns=trial_idx_test)

        # Calculate accuracy values for different number of selected channels
        accuracy_train, accuracy_test = [], []
        for i in range(1, self.num_channels + 1):
            accuracy_train.append(self.evaluate(x_train, y_train, num_selected_channels=i))
            accuracy_test.append(self.evaluate(x_test, y_test, num_selected_channels=i))
        self.optimum_number_of_channels = np.argmax(accuracy_train) + 1

        # Create a history dictionary to store the accuracy values
        history = {
            'single_channel_all_accuracy': single_channel_all_accuracy,
            'best_features_per_channel_train': best_features_per_channel_train,
            'selected_classifier': self.selected_classifier,
            'selected_channels_mean_accuracy': selected_channels_mean_accuracy,
            'selected_channels_std_accuracy': selected_channels_std_accuracy,
            'selected_classifier_train_accuracy': selected_classifier_train_accuracy,
            'selected_classifier_test_accuracy': selected_classifier_test_accuracy,
            'combination_mean_acc_train': accuracy_train,
            'combination_mean_acc_test': accuracy_test
        }

        df_result = pd.DataFrame({'Channel Name': selected_channel_names,
                                  'K-Fold Accuracy mean': [acc[1] for acc in single_channel_all_accuracy],
                                  'K-Fold Accuracy std': [acc[2] for acc in single_channel_all_accuracy],
                                  'Selected length': best_features_per_channel_train,
                                  'Train Accuracy': selected_classifier_train_accuracy,
                                  'Test Accuracy': selected_classifier_train_accuracy,
                                  'Combination Accuracy train': accuracy_train,
                                  'Combination Accuracy test': accuracy_test})

        df_result = df_result.set_index('Channel Name')
        #df_result = df_result.join(error_analysis)

        return history, df_result

    def predict(self, x_test, num_selected_channels=None):
        if num_selected_channels is None:
            num_selected_channels = x_test.shape[1]
        combined_classifier_output = []
        for clf, ch, s in zip(self.selected_classifier[:num_selected_channels],
                              self.selected_channels[:num_selected_channels],
                              self.selected_length[:num_selected_channels]):
            x_channel = x_test[:, ch, :s]
            y_val_pred = self.get_single_clf_output(clf, data=x_channel)
            combined_classifier_output.append(y_val_pred)
        prediction_all_cls = np.stack(combined_classifier_output, axis=0)
        prediction = self.get_combined_results(prediction_all_cls)
        prediction_probs = self.get_combined_probs(prediction_all_cls)
        return prediction, prediction_probs, combined_classifier_output

    def evaluate(self, x_test, y_test, num_selected_channels=None, metric='accuracy'):
        """
        Evaluates the performance of the classifier on the given test data.

        Args:
            x_test (numpy.ndarray): The input test data.
            y_test (numpy.ndarray): The target test data.
            num_selected_channels (int, optional): The number of selected channels to consider. If None,
                all channels will be used.

        Returns:
            float: The accuracy score of the classifier on the test data.

        """
        if num_selected_channels is None:
            num_selected_channels = self.optimum_number_of_channels
        prediction, prediction_probs, *_ = self.predict(x_test, num_selected_channels=num_selected_channels)
        result = {'number_of_channels': num_selected_channels,
                  'accuracy': accuracy_score(y_true=y_test, y_pred=prediction),
                  'f1': f1_score(y_true=y_test, y_pred=prediction, average='weighted'),
                  'precision': precision_score(y_true=y_test, y_pred=prediction, average='weighted'),
                  'recall': recall_score(y_true=y_test, y_pred=prediction, average='weighted')}
        if metric.lower() in list(result.keys()):
            return result[metric]
        else:
            return result

    def evaluate_combination_curve(self, x_test, y_test, num_selected_channels=None):
        """
        Evaluates the performance of the classifier on the given test data.

        Args:
            x_test (numpy.ndarray): The input test data.
            y_test (numpy.ndarray): The target test data.
            num_selected_channels (int, optional): The number of selected channels to consider. If None,
                all channels will be used.

        Returns:
            float: The accuracy score of the classifier on the test data.

        """
        accuracy_score_list, f_measure_score_list, precision_list, recall_list = [], [], [], []

        for i in range(x_test.shape[1]):
            prediction, *_ = self.predict(x_test, num_selected_channels=i + 1)
            accuracy_score_list.append(accuracy_score(y_true=y_test, y_pred=prediction))
            f_measure_score_list.append(f1_score(y_true=y_test, y_pred=prediction, average='weighted'))
            recall_list.append(recall_score(y_true=y_test, y_pred=prediction, average='weighted'))
            precision_list.append(precision_score(y_true=y_test, y_pred=prediction, average='weighted'))

        best_idx = np.argmax(accuracy_score_list)
        best_result = {
            'number_of_channels': best_idx + 1,
            'accuracy': accuracy_score_list[best_idx],
            'f1': accuracy_score_list[best_idx],
            'recall': recall_list[best_idx],
            'precision': precision_list[best_idx]
        }
        return accuracy_score_list, f_measure_score_list, recall_list, precision_list, best_result

    def build_clf_model(self):
        """
        Builds and returns a classifier model based on the specified classifier name.

        Returns:
            object: The built classifier model.

        Raises:
            ValueError: If the specified classifier is not implemented.

        """
        if self.classifier_name.lower() == 'bayesian_model':
            clf = BaysianModel()
        elif self.classifier_name.lower() in ['svc']:
            clf = SVC()
        elif self.classifier_name.lower() == 'naive_bayes':
            clf = GaussianNB()
        elif self.classifier_name.lower() == 'logistic_regression':
            clf = LogisticRegression()
        elif self.classifier_name.lower() == 'random_forest':
            clf = RandomForestClassifier(max_depth=2, n_estimators=5)
        elif self.classifier_name.lower() == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=(5,))
        elif self.classifier_name.lower() == 'lstm':
            clf = LSTMClassifier(input_size=1, hidden_size=5, num_classes=2)
        else:
            raise ValueError("Classifier not implemented")

        return clf

    def save_model(self, save_path):
        """
        Save the trained model
        Args:
            save_path:

        Returns:

        """
        saved_model = {
            'selected_channels': self.selected_channels,
            'selected_length': self.selected_length,
            'classifier_name': self.classifier_name,
            'selected_classifier': self.selected_classifier,
            'trial_idx_train': self.trial_idx_train,
            'optimum_number_of_channels': self.optimum_number_of_channels
        }

        with open(save_path, 'wb') as output:
            pickle.dump(saved_model, output)

    def load_model_and_evaluate(self, model_path):
        """
        Load the saved model
        Args:
            model_path:

        Returns:

        """
        with open(model_path, 'rb') as mdl:
            recovered_model = pickle.load(mdl)
        self.selected_channels = recovered_model['selected_channels']
        self.selected_length = recovered_model['selected_length']
        self.classifier_name = recovered_model['classifier_name']
        self.selected_classifier = recovered_model['selected_classifier']
        self.trial_idx_train = recovered_model['trial_idx_train']
        self.optimum_number_of_channels = recovered_model['optimum_number_of_channels']

    def get_best_features_per_channel(self, x_train, y_train, trial_idx_train, inner_cv, outer_fold):
        """
        Finds the best features for each channel of the input data using nested cross-validation.

        Args:
            x_train (numpy.ndarray): The input training data.
            y_train (numpy.ndarray): The target training data.
            trial_idx_train (numpy.ndarray): The index training data.
            inner_cv (scikit-learn CV splitter): The inner cross-validation splitter.
            outer_fold (int): The current outer fold index.

        Returns:
            tuple: A tuple containing the best number of features per channel and the accuracies for each channel.

        """
        best_features_per_channel, single_channel_all_accuracy = [], []

        for channel in tqdm(range(x_train.shape[1]),
                            desc='Find best features for each channel: Fold {}'.format(outer_fold)):
            single_channel_accuracies = []

            for n_features in range(1, x_train.shape[2] + 1):
                inner_accuracies = []
                y_val_pred_list, y_val_list = [], []
                trial_idx_valid_list = []
                for train_inner_index, val_index in inner_cv.split(x_train):
                    # Split the data for inner cross-validation
                    x_train_inner = x_train[train_inner_index][:, channel, :n_features]
                    x_val = x_train[val_index][:, channel, :n_features]
                    y_train_inner, y_val = y_train[train_inner_index], y_train[val_index]
                    if isinstance(trial_idx_train, pd.Series):
                        trial_idx_valid = trial_idx_train.values[val_index]
                    else:
                        trial_idx_valid = trial_idx_train[val_index]

                    # Build and train the classifier
                    clf = self.build_clf_model()
                    clf.fit(x_train_inner, np.squeeze(y_train_inner))

                    # Predict and calculate accuracy for validation data
                    y_val_pred = clf.predict(x_val)
                    inner_accuracies.append(accuracy_score(y_val, y_val_pred))
                    if self.error_analysis_mode is True:
                        y_val_pred_list.append(clf.predict_proba(x_val)[:, 1])
                    y_val_list.append(y_val)
                    trial_idx_valid_list.append(trial_idx_valid)

                single_channel_accuracies.append(
                    (n_features, np.mean(inner_accuracies), np.std(inner_accuracies),
                     np.concatenate([np.round(np.squeeze(np.abs(y_val_pred_list[i]) - np.squeeze(y_val_list[i])), 2)
                                     for i in range(len(y_val_list))], axis=0),
                     np.concatenate(trial_idx_valid_list, axis=0)))

            # Find the best number of features for the current channel
            mean_accuracy = np.round(
                np.array([single_channel_accuracy[1] for single_channel_accuracy in single_channel_accuracies]), 2)
            std_accuracy = np.array(
                [single_channel_accuracy[2] for single_channel_accuracy in single_channel_accuracies])
            indices = np.lexsort((-std_accuracy, mean_accuracy))
            best_index = indices[-1]

            best_n_features = single_channel_accuracies[best_index][0]
            # max(single_channel_accuracies, key=lambda x: x[1])[0]
            best_features_per_channel.append(int(best_n_features))
            single_channel_all_accuracy.append(single_channel_accuracies[int(best_n_features) - 1])

        # Save the best features per channel for the current outer fold
        df_fold_report = pd.DataFrame(single_channel_all_accuracy,
                                      columns=["Optimum Time", "Mean Accuracy", "Std Accuracy", "Values",
                                               "Column Names"])
        expanded_values = pd.DataFrame(df_fold_report["Values"].tolist(),
                                       columns=df_fold_report["Column Names"].iloc[0])
        # Concatenate the expanded values with the original DataFrame
        df_fold_report = pd.concat([df_fold_report.iloc[:, :4], expanded_values], axis=1)
        df_fold_report.insert(0, "Channel index", df_fold_report.index)

        self.best_time_per_channel['outer fold' + str(outer_fold)] = best_features_per_channel
        self.report_optimum_time['outer fold' + str(outer_fold)] = df_fold_report

        return best_features_per_channel, single_channel_all_accuracy

    def get_outputs_in_nested_kfold(self, x_train, y_train, trial_idx_train, n_splits_outer=None, n_splits_inner=None,
                                    outer_cv=None, inner_cv=None):
        """
        Performs nested k-fold cross-validation to obtain outputs and predicted labels for each channel.

        Args:
            trial_idx_train:
            inner_cv:
            outer_cv:
            x_train (numpy.ndarray): The input training data.
            y_train (numpy.ndarray): The target training data.
            n_splits_outer (int): The number of splits for the outer cross-validation.
            n_splits_inner (int): The number of splits for the inner cross-validation.

        Returns:
            tuple: A tuple containing the best number of features per channel for each fold, the predicted outputs per
                channel for each fold, and the true labels per channel for each fold.

        """
        if inner_cv is None or outer_cv is None:
            outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
            inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

        best_features_per_channel_all_folds = []
        outputs_per_channel_all_folds = []
        y_true_per_channel_all_folds = []

        for outer_fold_idx, (train_index, test_index) in enumerate(outer_cv.split(x_train)):
            print(outer_fold_idx)
            # Split data into outer train and validation sets
            x_train_outer, x_valid_outer = x_train[train_index], x_train[test_index]
            y_train_outer, y_valid_outer = y_train[train_index], y_train[test_index]
            trial_idx_train_outer, trial_idx_valid_outer = trial_idx_train.values[train_index], \
                trial_idx_train.values[test_index]
            # Find the optimum length of features per channel
            best_features_per_channel_per_fold, *_ = self.get_best_features_per_channel(x_train=x_train_outer,
                                                                                        y_train=y_train_outer,
                                                                                        trial_idx_train=trial_idx_train_outer,
                                                                                        inner_cv=inner_cv,
                                                                                        outer_fold=outer_fold_idx)
            best_features_per_channel_all_folds.append(best_features_per_channel_per_fold)

            # Calculate the output for the current fold
            outputs_per_channel_per_fold = []
            y_true_per_channel_per_fold = []
            probs_per_channel_per_fold = []
            for ch in range(x_train_outer.shape[1]):
                # Select the relevant features for the current channel
                x_channel_train = x_train_outer[:, ch, :best_features_per_channel_per_fold[ch]]
                x_channel_valid = x_valid_outer[:, ch, :best_features_per_channel_per_fold[ch]]

                # Build and train the classifier for the current channel
                clf = self.build_clf_model()
                clf.fit(x_channel_train, np.squeeze(y_train_outer))
                y_val_pred = self.get_single_clf_output(clf, data=x_channel_valid)

                outputs_per_channel_per_fold.append(y_val_pred)
                y_true_per_channel_per_fold.append(y_valid_outer)

                if self.error_analysis_mode is True:
                    y_val_prob = clf.predict_proba(x_channel_valid)
                    probs_per_channel_per_fold.append(y_val_prob[:, 1])

            outputs_per_channel_all_folds.append(outputs_per_channel_per_fold)
            y_true_per_channel_all_folds.append(y_true_per_channel_per_fold)

            if self.error_analysis_mode is True:
                self.report_model_error[f'outer fold {outer_fold_idx}'] = pd.DataFrame(np.round(np.abs(
                    np.squeeze(np.stack(y_true_per_channel_per_fold)) -
                    np.squeeze(np.stack(probs_per_channel_per_fold))), 2),
                    columns=trial_idx_valid_outer)
                self.report_model_error[f'outer fold {outer_fold_idx}'].insert(
                    0, "Channel index", self.report_model_error[f'outer fold {outer_fold_idx}'].index)

        return best_features_per_channel_all_folds, outputs_per_channel_all_folds, y_true_per_channel_all_folds

    def combine_channels(self, x_train, y_train, trial_idx_train, n_splits_outer=None, n_splits_inner=None,
                         outer_cv=None, inner_cv=None, verbose=True):
        """
        Combines channels to improve the classification accuracy using a nested k-fold cross-validation approach.

        Args:
            x_train (numpy.ndarray): The input training data.
            y_train (numpy.ndarray): The target training data.
            n_splits_outer (int): The number of splits for the outer cross-validation.
            n_splits_inner (int): The number of splits for the inner cross-validation.
            verbose (bool): If True, prints the selected channel and accuracy changes during the process.

        Returns:
            tuple: A tuple containing the selected channels, the mean accuracy for each selected channel,
                the standard deviation of accuracy for each selected channel, and the best number of features per channel
                for each fold.

        """
        # Obtain outputs and true labels per channel using nested k-fold cross-validation
        best_features_per_channel_all_folds, outputs_per_channel_all_folds, y_true_per_channel_all_folds = \
            self.get_outputs_in_nested_kfold(x_train, y_train, trial_idx_train, n_splits_outer=n_splits_outer,
                                             n_splits_inner=n_splits_inner, outer_cv=outer_cv, inner_cv=inner_cv)

        # Convert the lists of outputs and true labels per channel to numpy arrays
        outputs_per_channel_all_folds = [np.stack(output_fold, axis=0) for output_fold in outputs_per_channel_all_folds]
        y_true_per_channel_all_folds = [np.squeeze(np.stack(output_true, axis=0)) for output_true in
                                        y_true_per_channel_all_folds]

        channels_list = list(range(self.num_channels))
        selected_channels = []
        selected_channels_mean_accuracy, selected_channels_std_accuracy = [], []
        old_accuracy = 50

        for i in range(self.num_channels):
            mean_accuracy_for_new_candidates, std_accuracy_for_new_candidates = [], []

            for channel in channels_list:
                selected_channels_with_new_candidate = selected_channels.copy()
                selected_channels_with_new_candidate.append(channel)

                # Combine classifier outputs for the selected channels
                combined_classifier_output_new = [output_fold[selected_channels_with_new_candidate]
                                                  for output_fold in outputs_per_channel_all_folds]

                # Get combined predictions for the validation data
                y_val_pred_combined = [self.get_combined_results(combined_output_fold_new)
                                       for combined_output_fold_new in combined_classifier_output_new]

                # Calculate accuracy for the combined predictions
                combined_classifier_accuracy_new = [np.mean((true_output[0] == combined_output), axis=0)
                                                    for combined_output, true_output in
                                                    zip(y_val_pred_combined, y_true_per_channel_all_folds)]

                mean_accuracy_for_new_candidates.append(np.mean(combined_classifier_accuracy_new))
                std_accuracy_for_new_candidates.append(np.std(combined_classifier_accuracy_new))

            # Find the best channel based on mean accuracy and lowest standard deviation
            max_mean_accuracy = np.argmax(mean_accuracy_for_new_candidates)
            max_mask = (mean_accuracy_for_new_candidates == mean_accuracy_for_new_candidates[max_mean_accuracy])
            max_std = np.array(std_accuracy_for_new_candidates)[max_mask]
            min_std_idx = np.argmin(max_std)
            best_channel_index = np.where(max_mask)[0][min_std_idx]
            best_channel = channels_list.pop(best_channel_index)
            new_accuracy = mean_accuracy_for_new_candidates[best_channel_index]
            new_std = std_accuracy_for_new_candidates[best_channel_index]

            # Save the selected channel and its accuracy in the respective lists
            selected_channels.append(best_channel)
            selected_channels_mean_accuracy.append(new_accuracy)
            selected_channels_std_accuracy.append(new_std)

            # Print the selected channel and accuracy change if verbose is True
            if verbose is True:
                print("Selected channel is {}: Accuracy changed from {} to {}".format(best_channel, old_accuracy,
                                                                                      new_accuracy))
            old_accuracy = new_accuracy

        return selected_channels, selected_channels_mean_accuracy, \
            selected_channels_std_accuracy, best_features_per_channel_all_folds

    def get_combined_results(self, prediction):
        """
            Combines multiple predictions using a specified combination method.

            Args:
                prediction (numpy.ndarray): An array of predictions.

            Returns:
                numpy.ndarray: The combined prediction.

            Raises:
                ValueError: If the combination method is not implemented.

        """
        if self.combination_mode.lower() == 'maxvote':
            prediction_prob = np.mean(prediction, axis=0)
            prediction = 1 * (prediction_prob > 0.5)
        elif self.combination_mode.lower() == 'pol':
            prediction_prob = np.product(prediction / np.sum(prediction, axis=-1, keepdims=True), axis=0)
            prediction = np.argmax(prediction_prob, axis=-1)
        else:
            raise ValueError("Combination method {} Not implemented".format(self.combination_mode))

        return prediction

    def get_combined_probs(self, prediction):
        """
            Combines multiple predictions using a specified combination method.

            Args:
                prediction (numpy.ndarray): An array of predictions.

            Returns:
                numpy.ndarray: The combined prediction.

            Raises:
                ValueError: If the combination method is not implemented.

        """
        if self.combination_mode.lower() == 'maxvote':
            prediction_prob = np.mean(prediction, axis=0)
            prediction = 1 * (prediction_prob > 0.5)
        elif self.combination_mode.lower() == 'pol':
            prediction_prob = np.product(prediction / np.sum(prediction, axis=-1, keepdims=True), axis=0)[:, 1]
            prediction = np.argmax(prediction_prob, axis=-1)
        else:
            raise ValueError("Combination method {} Not implemented".format(self.combination_mode))

        return prediction_prob

    def get_single_clf_output(self, clf, data):
        """
            Retrieves the output of a single classifier for the given data, using the specified combination method.

            Args:
                clf: The classifier model.
                data: The single channel input data for prediction.

            Returns:
                numpy.ndarray: The prediction output of the classifier.

            Raises:
                ValueError: If the combination method is not implemented.

        """
        if self.combination_mode.lower() == 'maxvote':
            prediction = clf.predict(data)
        elif self.combination_mode.lower() == 'pol':
            prediction = clf.predict_proba(data)
        else:
            raise ValueError("Combination method {} Not implemented".format(self.combination_mode))
        return prediction
