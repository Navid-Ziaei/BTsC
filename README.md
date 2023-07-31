[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2307.15672)

# Bayesian Time-series Classification (BTsC) Model
Welcome to the GitHub repository for our BTsC model! This model is designed to decode visual stimuli from across the brain at a high temporal resolution by training on low-frequency and high-frequency power dynamics following image onset.
<br/>

## Table of Contents
* [General Information](#general-information)
* [Getting Started](#getting-started)
* [Example](#example)
* [Reading in Data](#reading-in-edf-data)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
* [Status](#status)
* [References](#references)
* [Contact](#contact)
<br/>

## General Information
Understanding how external stimuli are encoded in distributed neural activity is of significant interest in clinical and basic neuroscience. To address this need, it is essential to develop analytical tools capable of handling limited data and the intrinsic stochasticity present in neural data. In this study, we propose a straightforward Bayesian time series classifier (BTsC) model that tackles these challenges whilst maintaining a high level of interpretability.
<sup><sub>Navid Ziaei, Reza Saadatifard, Ali Yousefi, Behzad Nazari, Sydney S. Cash, Angelique C. Paulk. Bayesian Time-Series Classifier for Decoding Simple Visual Stimuli from Intracranial Neural Activity. 2023 August. </sup></sub>
<br/>

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. (See `requirements.txt` for details)

3. Prepare your dataset

4. Create the `./configs/settings.json` according to `./cinfigs/settings_sample.json`

5. Create the `./configs/device_path.json` according to `./cinfigs/device_path_sample.json`

6. Run the `main.py` script to execute the BTsC model.

In this example we create some chirp data and run the multitaper spectrogram on it.
```
from src.data import EEGDataLoader
from src.features import FeatureExtractor
from src.models import BTsCModel, find_best_time, train_and_evaluate_model_kfold
from src.settings import Settings, Paths

# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

# load raw data
data_loader = EEGDataLoader(paths, settings)
dataset = data_loader.load_data()

# Extract Features
feature_extractor = FeatureExtractor(dataset=dataset[1], settings=settings, feature_path=paths.feature_path)
time_features, features, labels, feature_name = feature_extractor.extract_BTsC_feature()
trial_idx = dataset[1].trial_info['Trial number']

# Find Best Time
# find_best_time(features, labels, trial_idx, feature_name, time_features, settings, paths)

# Train Model
model = BTsCModel(settings)
results, best_results, df_error_analysis = train_and_evaluate_model_kfold(model, features, labels, trial_idx,                                                                      feature_name, settings, paths)
```
<br/>

## Reading in Data
To load data, you need to put data in specific format. 
1. Define the dataset_name, patient, 
2. Put Your dataset in the following format
`Path to dataset/preared_dataset/task_name/patient_name/block_file_name.format`
<br/>

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run the BTsC model.

- `/data`: Contains scripts for data loading (`data_loader.py`, `clear_data_loader.py`) and preprocessing (`preprocess.py`).

- `/evaluators`: Contains the `evaluation.py` script for model performance evaluation.

- `/experiments`: Contains scripts for different experiments, such as `imagine_task.py`, `m_sequence_task.py`, and a script (`run_all_patients.py`) to run all patients' data.

- `/features`: Contains `feature_extraction.py` for feature extraction tasks.

- `/models`: Contains the main BTsC model (`btsc_model.py`), other classifier models (`classifier_models.py`), and model utilities (`model_utils.py`).

- `/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/utils`: Contains utility scripts (`utils.py`, `connectivity.py`) and a `multitapper` subfolder with multitaper spectrogram implementation (`multitaper_spectrogram_python.py`) and an example (`example.py`). 

- `/visualization`: Contains the `vizualize_utils.py` script for data and result visualization.
<br/>
- 
## Citations
The code contained in this repository for BTsC is companion to the paper:  

```
@InProceedings{navidziaei2023,
  title = {Bayesian Time-Series Classifier for Decoding Simple Visual Stimuli from Intracranial Neural Activity},
  author = {Navid Ziaei, Reza Saadatifard, Ali Yousefi, Behzad Nazari, Sydney S. Cash, Angelique C. Paulk},
  url = {https://arxiv.org/abs/2307.15672},
  booktitle = {Brain Informatics},
  year = {2023},
}
```
which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to BTsC! Please check out the [Contributing to BTsC guide](CONTRIBUTING.md) for guidelines about how to proceed.

## License

This project is licensed under the terms of the MIT license.
