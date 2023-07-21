from abc import ABC, abstractmethod
from src.visualization import plot_result_through_time, plot_channel_combination_accuracy, plot_single_channel_accuracy
from src.utils import append_dict_to_excel
from src.models import BTsCModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ModelEvaluator(ABC):
    def __init__(self, model, features, labels, trial_idx, feature_name, settings, paths):
        self.model = model
        self.features = features
        self.labels = labels
        self.trial_idx = trial_idx
        self.feature_name = feature_name
        self.settings = settings
        self.paths = paths
        self.history_list = []
        self.results = []
        self.best_results = []

    @abstractmethod
    def split_data(self):
        pass

    def save_reports(self, path):
        for key in self.model.report_optimum_time.keys():
            self.model.report_optimum_time[key].to_csv(path + f'report_optimum_time{key}.csv', index=False)
        for key in self.model.report_model_error.keys():
            self.model.report_model_error[key].to_csv(path + f'report_single_channel_error_{key}.csv', index=False)

    def train_model(self, x_train, y_train, trial_idx_train, x_test, y_test, trial_idx_test, path):
        history, df_result = self.model.fit(x_train, y_train, trial_idx_train, x_test=x_test, y_test=y_test,
                                            trial_idx_test=trial_idx_test, n_splits_outer=5, n_splits_inner=5,
                                            channel_names=self.feature_name)
        self.history_list.append(history)
        df_result.to_csv(path + 'results_and_error_analysis.csv')
        self.model.save_model(path + 'model_t{}.pkl'.format(x_train.shape[-1]))
        return history

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, metric='all')

    def train_and_evaluate(self):
        raise NotImplementedError("This function should be implemented in the concrete classes.")


class SingleSplitModelEvaluator(ModelEvaluator):
    def split_data(self):
        return train_test_split(self.features, self.labels, self.trial_idx, test_size=self.settings.test_size,
                                random_state=42, stratify=self.labels)

    def train_and_evaluate(self):
        x_train, x_test, y_train, y_test, trial_idx_train, trial_idx_test = self.split_data()
        history = self.train_model(x_train, y_train, trial_idx_train, x_test, y_test, trial_idx_test,
                                   self.paths.path_result_updated[0])
        # More operations can be added here, e.g., plotting

        test_result = self.evaluate_model(x_test, y_test)
        self.results.append(test_result)
        # Continue with the other operations...


class KFoldModelEvaluator(ModelEvaluator):
    def split_data(self):
        kf = KFold(n_splits=self.settings.num_fold, shuffle=True, random_state=42)
        return kf.split(self.features)

    def train_and_evaluate(self):
        splits = self.split_data()
        for fold_idx, (train_index, test_index) in enumerate(splits):
            print("\nTime {} Fold {} :".format(self.features.shape[-1], fold_idx))
            x_train, x_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            trial_idx_train, trial_idx_test = self.trial_idx.loc[train_index], self.trial_idx.loc[test_index]

            history = self.train_model(x_train, y_train, trial_idx_train, x_test, y_test, trial_idx_test,
                                       self.paths.path_result_updated[fold_idx + 1])
            # More operations can be added here, e.g., plotting

            test_result = self.evaluate_model(x_test, y_test)
            self.results.append(test_result)
            # Continue with the other operations...
