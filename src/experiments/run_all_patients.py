from src.data import *
from src.features import *
from src.models import *
from src.settings import *

patient_list = ['p01', 'p02', 'p03', 'p05']
experiment_list = [1, 1, 4, 2]
feature_types_list = ['combine']
classifiers_list = ['bayesian_model']
combination_mode_list = ['pol']

settings = Settings(target_class='color', task='flicker')
for classifier_type in classifiers_list:
    for feature_type in feature_types_list:
        for comb_type in combination_mode_list:
            for experiment, patient in zip(experiment_list, patient_list):
                plt.close('all')

                settings = Settings(verbose=0)
                settings.load_settings()
                settings.combination_mode = comb_type
                settings.classifier_name = classifier_type
                settings.feature_type = feature_type
                settings.patient = patient

                print("Patient: {} Feature type: {} Combination method: {} classifer: {}".format(settings.patient,
                                                                                                 settings.feature_type,
                                                                                                 settings.combination_mode,
                                                                                                 settings.classifier_name))

                # load device paths from device.json and create result paths
                paths = Paths(settings)
                paths.load_device_paths()

                # load raw data
                data_loader = EEGDataLoader(paths, settings)
                dataset = data_loader.load_raw_data()

                # Extract Features
                feature_extractor = FeatureExtractor(dataset=dataset, settings=settings)
                time_features, features, labels, feature_name = feature_extractor.extract_BTsC_feature(
                    feature_path=paths.feature_path)
                trial_idx = dataset.trial_info['Trial number']

                # Find Best Time
                find_best_time(features, labels, trial_idx, feature_name, time_features, settings, paths)


