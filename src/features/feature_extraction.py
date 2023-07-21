import numpy as np
from scipy import signal
from src.data import PreProcessing
from pathlib import Path
import copy
import os


class FeatureExtractor:
    def __init__(self, dataset, settings, feature_path):
        self.feature_path = feature_path
        self.task = settings.task
        self.dataset_meta = dataset.meta
        self.patient = settings.patient
        self.dataset = dataset
        self.fs = dataset.fs
        self.label = dataset.label
        self.load_features = settings.load_feature
        self.save_features = settings.save_features
        self.channel_name = dataset.channel_name
        self.label_name = ['Black', 'White']
        self.feature_type = settings.feature_type

        self.feature_list = dict()

    def extract_BTsC_feature(self):
        folder_path = os.path.join(self.feature_path, self.dataset_meta[0], self.dataset_meta[1])
        file_path = os.path.join(folder_path, self.dataset_meta[2][:-4])
        if self.load_features is True:
            print("Loading Pre Calculated Features ...")

            # TODO: make settings.feature_type a list of features and write a for loop here
            if os.path.exists(file_path + '_ERP.npy'):
                erp_features = np.load(file_path + '_ERP.npy')
                hgp_features = np.load(file_path + '_HGP.npy')
                labels = np.load(file_path + '_labels.npy')
                time_features = np.load(file_path + '_time_features.npy')
            else:
                raise ValueError(f"Features {file_path + '_ERP.npy'} doesn't exist")
        else:
            print("Extracting Features ...")
            """dataviz = DataVisualizer(self.dataset)
            dataviz.plot_power_spectrum(self.dataset.data, 0, 0, 0, 1, enable_plot=True, use_log=False)"""

            preprocessor = PreProcessing(dataset=copy.deepcopy(self.dataset))
            preprocessor.remove_line_noise(freqs_to_remove=[60],
                                           filter_length='auto',
                                           phase='zero',
                                           method='iir',
                                           trans_bandwidth=3.0,
                                           mt_bandwidth=None)
            preprocessor.filter_data_mne(low_cutoff=0.1, high_cutoff=250)
            preprocessed_dataset = preprocessor.dataset

            """dataviz = DataVisualizer(preprocessor.dataset)
            dataviz.plot_power_spectrum(preprocessor.dataset.data, 0, 0, 0, 1, enable_plot=True, use_log=False)
            plt.show()"""

            if self.task == 'm_sequence':
                t_max = 0.8
            else:
                t_max = 1
            time_features, erp_features = self.extract_ERP_features(t_min=0, t_max=t_max)
            hgp_features = self.extract_HGP_features(t_min=0, t_max=t_max,
                                                     window_size=time_features[1] - time_features[0])
            labels = preprocessed_dataset.label.values.astype('int')
            if self.save_features is True:
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                np.save(file_path + '_time_features.npy', time_features)
                np.save(file_path + '_ERP.npy', erp_features)
                np.save(file_path + '_HGP.npy', hgp_features)
                np.save(file_path + '_labels.npy', preprocessed_dataset.label)

        # combine features
        if self.feature_type == 'erp':
            features = erp_features / np.max(erp_features, axis=(0, 2), keepdims=True)
            feature_name = [ch + '_ERP' for ch in self.channel_name]
        elif self.feature_type == 'hgp':
            features = hgp_features[:, :, :-1] / np.max(hgp_features[:, :, :-1], axis=(0, 2), keepdims=True)
            feature_name = [ch + '_HGP' for ch in self.channel_name]
        else:
            erp_features = erp_features / np.max(erp_features, axis=(0, 2), keepdims=True)
            hgp_features = hgp_features / np.max(hgp_features, axis=(0, 2), keepdims=True)
            features = np.concatenate([erp_features, hgp_features[:, :, :-1]], axis=1)
            feature_name = [ch + '_ERP' for ch in self.channel_name]
            feature_name.extend([ch + '_HGP' for ch in self.channel_name])

        return time_features, features, labels, feature_name

    def extract_ERP_features(self, t_min=0, t_max=1):
        # Use lowpass filter
        # Filter data
        preprocessor = PreProcessing(dataset=copy.deepcopy(self.dataset))
        preprocessor.filter_data_mne(low_cutoff=None, high_cutoff=7)
        preprocessor.resample(f_resample=15)

        dataset = preprocessor.dataset

        # Convert time interval to sample indices
        idx_start = np.argmin(np.abs(dataset.time - t_min))
        idx_end = np.argmin(np.abs(dataset.time - t_max))

        # Extract data in the time interval
        features = dataset.data[..., idx_start:idx_end]
        time = dataset.time[idx_start:idx_end]

        self.feature_list['erp'] = features

        return time, features

    def extract_HGP_features(self, t_min=0, t_max=1, window_size=65):

        # Use lowpass filter
        # Filter data
        preprocessor = PreProcessing(dataset=copy.deepcopy(self.dataset))
        preprocessor.filter_data_mne(low_cutoff=65, high_cutoff=115)

        dataset = preprocessor.dataset

        # Convert time interval to sample indices
        idx_start = np.argmin(np.abs(dataset.time - t_min))
        idx_end = np.argmin(np.abs(dataset.time - t_max))

        # Extract data in the time interval
        processed_data = dataset.data[..., idx_start:idx_end]
        window_index = int(window_size * dataset.fs)
        time = np.arange(0, idx_end - idx_start, window_index)

        features = []
        for t_idx in list(time):
            features.append(
                np.log(np.sqrt(np.mean(processed_data[:, :, t_idx:t_idx + window_index] ** 2, axis=-1))))

        features = np.stack(features, axis=-1)

        return features
