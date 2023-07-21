from src.visualization import plot_result_through_time, plot_channel_combination_accuracy, plot_single_channel_accuracy
from src.utils import append_dict_to_excel
from .btsc_model import BTsCModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_best_time(features, labels, trial_idx, feature_name, time_features, settings, paths, time_idx=None):
    history_list = []
    # times
    if time_idx is None:
        time_idx = list(range(2, features.shape[-1]))
    for t in time_idx:
        print("\nTime = {}".format(t))
        model = BTsCModel(settings)
        results, best_results, _ = train_and_evaluate_model_kfold(model, features[:, :, :t], labels, trial_idx,
                                                                  feature_name, settings, paths)
        mean_result = {k: np.mean([d[k] for d in results]) for k in results[0]}
        std_result = {k: np.std([d[k] for d in results]) for k in results[0]}

        mean_best_result = {k: np.mean([d[k] for d in best_results]) for k in best_results[0]}
        std_best_result = {k: np.std([d[k] for d in best_results]) for k in best_results[0]}

        # best results are when we use the best number of channels based on test data and the other one on validation
        # data
        history = {
            'mean_result': mean_result,
            'std_result': std_result,
            'mean_best_result': mean_best_result,
            'std_best_result': std_best_result
        }
        history_list.append(history)

        with open(paths.path_result_updated[0] + 'results_history_time{}.json'.format(t), 'w') as output:
            json.dump(history, output, indent=4)

    number_of_channels_through_time = [h['mean_result']['number_of_channels'] for h in history_list]
    std_number_of_channels_through_time = [h['std_result']['number_of_channels'] for h in history_list]

    mean_accuracy_list_through_time = [h['mean_result']['accuracy'] for h in history_list]
    std_accuracy_list_through_time = [h['std_result']['accuracy'] for h in history_list]

    mean_f1_list_through_time = [h['mean_result']['f1'] for h in history_list]
    std_f1_list_through_time = [h['std_result']['f1'] for h in history_list]

    mean_best_accuracy_list_through_time = [h['mean_best_result']['accuracy'] for h in history_list]
    std_best_accuracy_list_through_time = [h['std_best_result']['accuracy'] for h in history_list]

    mean_best_f1_list_through_time = [h['mean_best_result']['f1'] for h in history_list]
    std_best_f1_list_through_time = [h['std_best_result']['f1'] for h in history_list]

    # TODO: seperated number of used HGP and ERP features

    plot_result_through_time(time_features, time_idx,
                             mean_accuracy_list_through_time=mean_accuracy_list_through_time,
                             std_accuracy_list_through_time=std_accuracy_list_through_time,
                             number_of_channels_through_time=number_of_channels_through_time,
                             number_of_channels_through_time_std=std_number_of_channels_through_time,
                             save_path=paths.path_result[0],
                             file_name='accuracy_through_time',
                             title="Results")

    plot_result_through_time(time_features, time_idx,
                             mean_accuracy_list_through_time=mean_f1_list_through_time,
                             std_accuracy_list_through_time=std_f1_list_through_time,
                             number_of_channels_through_time=number_of_channels_through_time,
                             number_of_channels_through_time_std=std_number_of_channels_through_time,
                             save_path=paths.path_result[0],
                             file_name='f1_through_time',
                             title="Results")

    plot_result_through_time(time_features, time_idx,
                             mean_accuracy_list_through_time=mean_best_accuracy_list_through_time,
                             std_accuracy_list_through_time=std_best_accuracy_list_through_time,
                             number_of_channels_through_time=number_of_channels_through_time,
                             number_of_channels_through_time_std=std_number_of_channels_through_time,
                             save_path=paths.path_result[0],
                             file_name='best_accuracy_through_time',
                             title="Best Results")

    plot_result_through_time(time_features, time_idx,
                             mean_accuracy_list_through_time=mean_best_f1_list_through_time,
                             std_accuracy_list_through_time=std_best_f1_list_through_time,
                             number_of_channels_through_time=number_of_channels_through_time,
                             number_of_channels_through_time_std=std_number_of_channels_through_time,
                             save_path=paths.path_result[0],
                             file_name='best_f1_through_time',
                             title="Best Results")

    best_accuracy_index = np.argmax(mean_accuracy_list_through_time)
    with open(paths.path_result[0] + 'results.txt', 'w') as f:
        f.write("Best Accuracy: {} +- {} \n".format(np.max(mean_accuracy_list_through_time),
                                                    std_accuracy_list_through_time[best_accuracy_index]))
        f.write("Time: {} \n".format(time_features[time_idx[best_accuracy_index]]))


def train_and_evaluate_model_kfold(model, features, labels, trial_idx, feature_name, settings, paths):
    """

    Args:
        model:
        features:
        labels:
        trial_idx:
        feature_name:
        settings:
        paths:

    Returns:

    """
    paths.update_path(features.shape[-1])
    results, best_results, history_list = [], [], []
    fold_names = ['fold' + str(i) for i in range(1, settings.num_fold + 1)]
    df_error_analysis = pd.DataFrame(None, index=trial_idx.values, columns=fold_names)
    if settings.num_fold == 1:
        if settings.load_pretrained_model is False:
            # Perform a train-test split
            x_train, x_test, y_train, y_test, trial_idx_train, trial_idx_test = train_test_split(features, labels,
                                                                                                 trial_idx,
                                                                                                 test_size=settings.test_size,
                                                                                                 random_state=42,
                                                                                                 stratify=labels)
            # Train the model on the current fold
            history, df_result = model.fit(x_train, y_train, trial_idx_train,
                                           x_test=x_test, y_test=y_test, trial_idx_test=trial_idx_test,
                                           n_splits_outer=5, n_splits_inner=5, channel_names=feature_name)

            for key in model.report_optimum_time.keys():
                model.report_optimum_time[key].to_csv(paths.path_result_updated[0] + f'report_optimum_time{key}.csv',
                                                      index=False)

            for key in model.report_model_error.keys():
                model.report_model_error[key].to_csv(
                    paths.path_result_updated[0] + f'report_single_channel_error_{key}.csv', index=False)

            history_list.append(history)
            df_result.to_csv(paths.path_result_updated[0] + 'results_and_error_analysis.csv')
            model.save_model(paths.path_result_updated[0] + 'model_t{}.pkl'.format(x_train.shape[-1]))

            single_channel_accuracy = np.stack(history['single_channel_all_accuracy'])
            plot_single_channel_accuracy(single_channel_accuracy[:, 1], single_channel_accuracy[:, 2],
                                         save_path=paths.path_result_updated[0],
                                         title="Single classifier accuracy",
                                         file_name='single_channel_accuracy.svg')
            fig, axs = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
            plot_channel_combination_accuracy(mean_accuracy=history['combination_mean_acc_train'], axs=axs, fig=fig,
                                              save_file=False)
            plot_channel_combination_accuracy(mean_accuracy=history['combination_mean_acc_test'], axs=axs, fig=fig,
                                              save_file=False)
            axs.legend(["Validation", "test"])
            fig.savefig(paths.path_result_updated[0] + "combination_accuracy.svg")
            fig.savefig(paths.path_result_updated[0] + "combination_accuracy.png")

            plot_channel_combination_accuracy(history['combination_mean_acc_train'],
                                              save_path=paths.path_result_updated[0],
                                              file_name='combination_accuracy_train', save_file=True)
            plot_channel_combination_accuracy(history['combination_mean_acc_test'],
                                              save_path=paths.path_result_updated[0],
                                              file_name='combination_accuracy_test', save_file=True)
        else:
            model.load_model_and_evaluate(
                paths.path_model[0] + 'model_t{}.pkl'.format(features.shape[-1]))
            all_indices = list(trial_idx.values)
            train_indices = list(model.trial_idx_train.values)
            indices = [all_indices.index(element) for element in all_indices if element not in train_indices]
            x_test, y_test, trial_idx_test = features[indices], labels[indices], trial_idx[indices]

        test_result = model.evaluate(x_test, y_test, metric='all')
        prediction, prediction_probs, *_ = model.predict(x_test, num_selected_channels=model.optimum_number_of_channels)
        df_error_analysis.loc[trial_idx_test] = np.squeeze(np.abs(prediction_probs - y_test))[:, np.newaxis]

        accuracy_score_list, f_measure_score_list, recall_list, precision_list, best_test_result = \
            model.evaluate_combination_curve(x_test, y_test)

        print("Test accuracy: {:.2f}%".format(test_result['accuracy'] * 100))
        results.append(test_result)
        best_results.append(best_test_result)
    else:
        # Perform k-fold cross-validation
        kf = KFold(n_splits=settings.num_fold, shuffle=True, random_state=42)

        for fold_idx, (train_index, test_index) in enumerate(kf.split(features)):
            print("\nTime {} Fold {} :".format(features.shape[-1], fold_idx))
            # Train the model on the current fold
            if settings.load_pretrained_model is False:
                x_train, x_test = features[train_index], features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                trial_idx_train, trial_idx_test = trial_idx.loc[train_index], trial_idx.loc[test_index]

                history, df_result = model.fit(x_train, y_train, trial_idx_train,
                                               x_test=x_test, y_test=y_test, trial_idx_test=trial_idx_test,
                                               n_splits_outer=5, n_splits_inner=5, channel_names=feature_name)

                for key in model.report_optimum_time.keys():
                    model.report_optimum_time[key].to_csv(
                        paths.path_result_updated[fold_idx + 1] + f'report_optimum_time{key}.csv', index=False)

                for key in model.report_model_error.keys():
                    model.report_model_error[key].to_csv(
                        paths.path_result_updated[fold_idx + 1] + f'report_single_channel_error_{key}.csv', index=False)

                history_list.append(history)
                model.save_model(
                    paths.path_model_updated[fold_idx + 1] + 'model_t{}_f{}.pkl'.format(x_train.shape[-1], fold_idx))
                df_result.to_csv(paths.path_result_updated[fold_idx + 1] + 'results_and_error_analysis.csv')

                single_channel_accuracy = np.stack(history['single_channel_all_accuracy'])
                plot_single_channel_accuracy(single_channel_accuracy[:, 1], single_channel_accuracy[:, 2],
                                             save_path=paths.path_result_updated[fold_idx + 1],
                                             title="Single classifier accuracy",
                                             file_name='single_channel_accuracy.svg')
                # TODO: append single channels test accuracy in a list then use plot_single_channel_accuracy to plot
                #  mean and std of single channel classifiers on test

                fig, axs = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
                plot_channel_combination_accuracy(mean_accuracy=history['combination_mean_acc_train'], axs=axs, fig=fig,
                                                  save_file=False)
                plot_channel_combination_accuracy(mean_accuracy=history['combination_mean_acc_test'], axs=axs, fig=fig,
                                                  save_file=False)
                axs.legend(["Validation", "test"])
                fig.savefig(paths.path_result_updated[fold_idx + 1] + "combination_accuracy.svg")
                fig.savefig(paths.path_result_updated[fold_idx + 1] + "combination_accuracy.png")
                plot_channel_combination_accuracy(history['combination_mean_acc_train'],
                                                  save_path=paths.path_result_updated[fold_idx + 1],
                                                  file_name='combination_accuracy_train', save_file=True)
                plot_channel_combination_accuracy(history['combination_mean_acc_test'],
                                                  save_path=paths.path_result_updated[fold_idx + 1],
                                                  file_name='combination_accuracy_test', save_file=True)

            else:
                model.load_model_and_evaluate(
                    paths.path_model_updated[fold_idx + 1] + 'model_t{}_f{}.pkl'.format(features.shape[-1], fold_idx))
                all_indices = list(trial_idx.values)
                train_indices = list(model.trial_idx_train.values)
                indices = [all_indices.index(element) for element in all_indices if element not in train_indices]
                x_test, y_test, trial_idx_test = features[indices], labels[indices], trial_idx[indices]

            test_result = model.evaluate(x_test, y_test, metric='all')

            accuracy_score_list, f_measure_score_list, recall_list, precision_list, best_test_result = \
                model.evaluate_combination_curve(x_test, y_test)
            prediction, prediction_probs, *_ = model.predict(x_test,
                                                             num_selected_channels=model.optimum_number_of_channels)
            df_error_analysis['fold' + str(fold_idx + 1)].loc[trial_idx_test] = np.squeeze(
                np.abs(prediction_probs - y_test))

            print("Test f1 Score for fold {}: {:.2f}%".format(fold_idx + 1, test_result['accuracy'] * 100))
            results.append(test_result)
            best_results.append(best_test_result)

            if len(history_list) > 0:
                combination_mean_acc_train_allfolds = np.stack([h['combination_mean_acc_train'] for h in history_list],
                                                               axis=-1)
                combination_mean_acc_test_allfolds = np.stack([h['combination_mean_acc_test'] for h in history_list],
                                                              axis=-1)

                fig, axs = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
                plot_channel_combination_accuracy(mean_accuracy=np.mean(combination_mean_acc_train_allfolds, axis=-1),
                                                  axs=axs, fig=fig,
                                                  save_file=False)
                plot_channel_combination_accuracy(mean_accuracy=np.mean(combination_mean_acc_test_allfolds, axis=-1),
                                                  axs=axs, fig=fig,
                                                  save_file=False)
                axs.legend(["Validation", "test"])
                fig.savefig(paths.path_result_updated[0] + "combination_accuracy_without_confidence.svg")
                fig.savefig(paths.path_result_updated[0] + "combination_accuracy_without_confidence.png")

                plot_channel_combination_accuracy(np.mean(combination_mean_acc_train_allfolds, axis=-1),
                                                  save_path=paths.path_result_updated[0],
                                                  file_name='combination_accuracy_train_all', save_file=True)
                plot_channel_combination_accuracy(np.mean(combination_mean_acc_test_allfolds, axis=-1),
                                                  save_path=paths.path_result_updated[0],
                                                  file_name='combination_accuracy_test_all', save_file=True)

                fig, axs = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
                plot_channel_combination_accuracy(mean_accuracy=np.mean(combination_mean_acc_train_allfolds, axis=-1),
                                                  std_accuracy=np.std(combination_mean_acc_train_allfolds, axis=-1),
                                                  axs=axs, fig=fig,
                                                  save_file=False)
                plot_channel_combination_accuracy(mean_accuracy=np.mean(combination_mean_acc_test_allfolds, axis=-1),
                                                  std_accuracy=np.std(combination_mean_acc_test_allfolds, axis=-1),
                                                  axs=axs, fig=fig,
                                                  save_file=False)
                axs.legend(["Validation", "test"])
                fig.savefig(paths.path_result_updated[0] + "combination_accuracy_with_confidence.svg")
                fig.savefig(paths.path_result_updated[0] + "combination_accuracy_with_confidence.png")

                plot_channel_combination_accuracy(np.mean(combination_mean_acc_train_allfolds, axis=-1),
                                                  std_accuracy=np.std(combination_mean_acc_train_allfolds, axis=-1),
                                                  save_path=paths.path_result_updated[0],
                                                  file_name='combination_accuracy_train_all', save_file=True)
                plot_channel_combination_accuracy(np.mean(combination_mean_acc_test_allfolds, axis=-1),
                                                  std_accuracy=np.std(combination_mean_acc_test_allfolds, axis=-1),
                                                  save_path=paths.path_result_updated[0],
                                                  file_name='combination_accuracy_test_all', save_file=True)

    df_error_analysis.to_csv(paths.path_result_updated[0] + "Error_analysis.csv")

    mean_result = {k: np.mean([d[k] for d in results]) for k in results[0]}
    std_result = {k: np.std([d[k] for d in results]) for k in results[0]}

    mean_best_result = {k: np.mean([d[k] for d in best_results]) for k in best_results[0]}
    std_best_result = {k: np.std([d[k] for d in best_results]) for k in best_results[0]}

    with open(paths.path_result_updated[0] + 'results.txt', 'w') as f:
        for k in mean_result.keys():
            f.write("The {} for test set is {} +- {} \n".format(k, mean_result[k], std_result[k]))

    # 2- we wnt to save the result in xlsx file
    res_dict = {'Patient': [settings.patient],
                'Task': [settings.task],
                'Target': [settings.target_class],
                'Features': [settings.feature_type],
                'Combination mode': [settings.combination_mode],
                'Classifier': [settings.classifier_name],
                'Folds': [settings.num_fold],
                'Time Limit': [features.shape[-1]],
                'Number of channels': [features.shape[1]],
                'Test Samples': [settings.test_size],
                'Best number of channel (valid)': [model.optimum_number_of_channels],
                'Result Path': [
                    '=HYPERLINK("{}","{}")'.format(paths.path_result_updated[0].replace('\\', '/').replace('//', '/'),
                                                   paths.folder_name)],
                'Dataset Folder name': [
                    '=HYPERLINK("{}","{}")'.format(paths.path_dataset_raw[0].replace('\\', '/').replace('//', '/'),
                                                   settings.patient)],
                'Dataset File name': ['File name']}

    mean_result = {k + '_mean': [v] for k, v in mean_result.items()}
    std_result = {k + '_std': [v] for k, v in std_result.items()}

    mean_best_result = {'best_' + k + '_mean': [v] for k, v in mean_best_result.items()}
    std_best_result = {'best_' + k + '_std': [v] for k, v in std_best_result.items()}

    res_dict.update(mean_result)
    res_dict.update(std_result)
    res_dict.update(mean_best_result)
    res_dict.update(std_best_result)

    while True:
        try:
            append_dict_to_excel(filename=paths.base_path + 'result_{}.csv'.format(settings.task), data_dict=res_dict)
            # If the file was successfully edited, break the loop
            break
        except PermissionError:
            print(
                f"The file {paths.base_path + 'result_{}.csv'.format(settings.task)} is open in another program. Please "
                f"close it to continue.")
            # Wait for 10 seconds before trying again
            time.sleep(10)

    return results, best_results, df_error_analysis
