import pandas as pd

from src.data import OCEDEEGDataLoader
from src.settings import Settings, Paths
from src.data import PreProcessing
from src.visualization import DataVisualizer, plot_gaussian
import copy
import matplotlib.pyplot as plt
import numpy as np

t_min, t_max = 0, 1
window_size = 0.06599999999999984

# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

# load raw data
data_loader = OCEDEEGDataLoader(paths, settings)
dataset = data_loader.load_data()

# ##################### Preprocess ###############################
preprocessor = PreProcessing(dataset=copy.deepcopy(dataset[-1]))
preprocessor.remove_line_noise(freqs_to_remove=[60],
                               filter_length='auto',
                               phase='zero',
                               method='iir',
                               trans_bandwidth=3.0,
                               mt_bandwidth=None)

preprocessed_data = preprocessor.filter_data_mne(low_cutoff=0.1, high_cutoff=250)
preprocessed_data = preprocessor.remove_base_line(baseline_t_min=-0.5, baseline_t_max=-0.05)

# ##################### HGP ###############################
preprocessor_hgp = PreProcessing(dataset=copy.deepcopy(preprocessor.dataset))
high_gamma_data = preprocessor_hgp.filter_data_mne(low_cutoff=65, high_cutoff=115)

# Convert time interval to sample indices
dataset = preprocessor_hgp.dataset
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

hgp_features = np.stack(features, axis=-1)

# ##################### ERP ###############################
ERP_preprocessor = PreProcessing(dataset=copy.deepcopy(preprocessor.dataset))
data_ERP = ERP_preprocessor.filter_data_mne(low_cutoff=None, high_cutoff=7)

DS_preprocessor = PreProcessing(dataset=copy.deepcopy(ERP_preprocessor.dataset))
DS_preprocessor.resample(f_resample=15)
dataset_ds = DS_preprocessor.dataset

# Convert time interval to sample indices
idx_start = np.argmin(np.abs(dataset_ds.time - t_min))
idx_end = np.argmin(np.abs(dataset_ds.time - t_max))

# Extract data in the time interval
features = dataset_ds.data[..., idx_start:idx_end]
time = dataset_ds.time[idx_start:idx_end]

erp_features = features

# ##################### Visualization ###############################

channel_idx = 18
trial_idx = 5


def plot_feature_extraction_procedure(channel_index, trial_index, t1, t2):
    if isinstance(dataset_ds.label, pd.DataFrame):
        labels = dataset_ds.label.values.astype(int)
    else:
        labels = dataset_ds.label.astype(int)
    fig, ax = plt.subplots(1, 1)
    data_visualizer1 = DataVisualizer(preprocessor.dataset)
    data_visualizer1.plot_single_channel_data(preprocessor.dataset.data, t_min=t_min, t_max=t_max,
                                              trial_idx=trial_index,
                                              channel_idx=channel_index, ax=ax)
    plt.show()

    data_visualizer2 = DataVisualizer(ERP_preprocessor.dataset)

    fig, ax = plt.subplots(1, 1)
    data_visualizer1.plot_single_channel_data(preprocessor.dataset.data, t_min=t_min, t_max=t_max,
                                              trial_idx=trial_index, ax=ax,
                                              channel_idx=channel_index, alpha=0.5)
    data_visualizer2.plot_single_channel_data(data_ERP, t_min=t_min, t_max=t_max, trial_idx=trial_index,
                                              channel_idx=channel_index, ax=ax, alpha=1, color='red')

    plt.show()

    fig, ax = plt.subplots(1, 1)
    data_visualizer2.plot_single_channel_data(data_ERP, t_min=t_min, t_max=t_max, trial_idx=trial_index,
                                              channel_idx=channel_index, ax=ax, alpha=0.5, color='green')
    ax.plot(time, erp_features[trial_index, channel_index, :], marker='o', color='red')
    ax.set_xlabel("Time (second)")
    ax.set_ylabel("Amplitude")
    ax.set_title(preprocessor.dataset.channel_name[channel_index] + ' Label = ' + settings.target_class[
        labels[trial_index]])
    plt.show()

    # hgp
    fig, ax = plt.subplots(1, 1)
    data_visualizer1.plot_single_channel_data(preprocessor.dataset.data, t_min=t_min, t_max=t_max,
                                              trial_idx=trial_index, ax=ax,
                                              channel_idx=channel_index, alpha=0.5)
    data_visualizer = DataVisualizer(preprocessor_hgp.dataset)
    data_visualizer.plot_single_channel_data(high_gamma_data, t_min=t_min, t_max=t_max, trial_idx=trial_index,
                                             channel_idx=channel_index, color='green')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(time, hgp_features[trial_index, channel_index, :len(time)], marker='o')
    ax.set_xlabel("Time (second)")
    ax.set_ylabel("Amplitude")
    ax.set_title(preprocessor.dataset.channel_name[channel_index] + ' Label = ' + settings.target_class[
        labels[trial_index]])
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plot_gaussian(erp_features, ch=channel_index, t1=t1, t2=t2, ax=ax, title="ERP Scatter Plot")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plot_gaussian(hgp_features, ch=channel_index, t1=t1, t2=t2, ax=ax, title="HGP Scatter Plot")
    plt.show()


plot_feature_extraction_procedure(118, 5, 3, 4)
