from src.data import iEEGDataLoader, OCEDEEGDataLoader
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
data_loader = OCEDEEGDataLoader(paths, settings)
dataset = data_loader.load_data()

# Extract Features
feature_extractor = FeatureExtractor(dataset=dataset[0], settings=settings, feature_path=paths.feature_path)
time_features, features, labels, feature_name = feature_extractor.extract_BTsC_feature()
trial_idx = dataset[0].trial_info['Trial number']

# Find Best Time
# find_best_time(features, labels, trial_idx, feature_name, time_features, settings, paths)

# Train Model
model = BTsCModel(settings)
results, best_results, df_error_analysis = train_and_evaluate_model_kfold(model, features, labels, trial_idx,
                                                                          feature_name, settings, paths)


