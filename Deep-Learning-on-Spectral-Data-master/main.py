import random
import tensorflow as tf
import pandas as pd

from constants import *
from utils import (
    make_all_root_dirs,
    read_NIR_data_combined,
    read_NIR_data,
    read_pharma_data,
    read_feature_list_from_csv,
    plot_keras_model,
)
from data_exploration import env_data_exploration, spectral_data_analysis
from training import train_classic_models, train_multivariate_models, train_ann_models
from transformer_training import train_transformer
from evaluation import evaluate_models
from preprocessing import filter_important_features, get_features_from_importances

make_all_root_dirs()
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
# TODO: Set random state in sklearn models

# Read and explore the combined NIR data and the training and test datasets
if False:
    env_features, spectral_features, target = read_NIR_data_combined(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True
    )
    env_data_exploration(
        env_features,
        COMBINED_NIR_PLOTS_ROOT,
        target=target,
        target_columns=[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT],
    )
    spectral_data_analysis(spectral_features, COMBINED_NIR_PLOTS_ROOT, PEAK_REGIONS)

    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True
    )
    env_data_exploration(
        training_data[0],
        TRAIN_NIR_PLOTS_ROOT,
        target=training_data[2],
        target_columns=[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT],
    )
    spectral_data_analysis(training_data[1], TRAIN_NIR_PLOTS_ROOT, PEAK_REGIONS)

    env_data_exploration(
        test_data[0],
        TEST_NIR_PLOTS_ROOT,
        target=test_data[2],
        target_columns=[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT],
    )
    spectral_data_analysis(test_data[1], TEST_NIR_PLOTS_ROOT, PEAK_REGIONS)

# Train classic machine learning models
if False:
    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True
    )

    train_classic_models(
        training_data[0], training_data[1], training_data[2], models_dir=CLASSIC_ML_DIR
    )

    evaluate_models(
        training_data,
        test_data,
        models_dir=CLASSIC_ML_DIR,
        metrics_output_file=CLASSIC_ML_METRICS_FILE,
        predictions_output_file=CLASSIC_ML_PREDICTIONS_FILE,
        plots_dir=CLASSIC_ML_PLOTS_DIR,
    )

# Train classic multivariate models
if False:
    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True, multivariate=True
    )

    train_multivariate_models(
        training_data[0],
        training_data[1],
        training_data[2],
        models_dir=CLASSIC_MULTIVAR_ML_DIR,
    )

    evaluate_models(
        training_data,
        test_data,
        models_dir=CLASSIC_MULTIVAR_ML_DIR,
        metrics_output_file=CLASSIC_MULTIVAR_ML_METRICS_FILE,
        predictions_output_file=CLASSIC_MULTIVAR_ML_PREDICTIONS_FILE,
        plots_dir=CLASSIC_MULTIVAR_ML_PLOTS_DIR,
    )

# Filter important features
if False:
    important_features = get_features_from_importances(
        filter_important_features(
            MODEL_NAMES[2],  # Using Lasso to filter features
            CLASSIC_MULTIVAR_ML_DIR,
            test_data,
            multivariate=True,
            threshold=75,  # Percentile threshold
            plot=False,
            exclude_narrow_bands=False,
        ),
        save_to_file=SELECTED_FEATURES_FILE,
    )

# Train multivariate ANN
if False:
    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True, multivariate=True
    )

    selected_features = read_feature_list_from_csv(SELECTED_FEATURES_FILE)

    train_ann_models(
        pd.concat([training_data[0], training_data[1]], axis=1)[selected_features],
        training_data[2],
        models_dir=ANN_MULTIVAR_DIR,
    )

    evaluate_models(
        training_data,
        test_data,
        models_dir=ANN_MULTIVAR_DIR,
        metrics_output_file=ANN_MULTIVAR_METRICS_FILE,
        predictions_output_file=ANN_MULTIVAR_PREDICTIONS_FILE,
        plots_dir=ANN_MULTIVAR_PLOTS_DIR,
        models_to_evaluate={"best_ANN": None},
        selected_features_file=SELECTED_FEATURES_FILE,
        model_type=KERAS_MODEL,
    )

    plot_keras_model("best_ANN", models_dir=ANN_MULTIVAR_DIR)

# Apply data augmentation
if False:
    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True, multivariate=True
    )
    # TODO: To be implemented


# Train a transformer model
if False:
    training_data, test_data = read_NIR_data(
        NIR_TRAINING_DATA_FILE, NIR_TEST_DATA_FILE, remove_nulls=True, multivariate=True
    )

    # selected_features = read_feature_list_from_csv(SELECTED_FEATURES_FILE)

    train_transformer(
        # pd.concat([training_data[0], training_data[1]], axis=1)[selected_features],
        pd.concat([training_data[0], training_data[1]], axis=1),
        training_data[2],
        models_dir=TRANSFORMER_MULTIVAR_DIR,
    )

    evaluate_models(
        training_data,
        test_data,
        models_dir=TRANSFORMER_MULTIVAR_DIR,
        metrics_output_file=TRANSFORMER_MULTIVAR_METRICS_FILE,
        predictions_output_file=TRANSFORMER_MULTIVAR_PREDICTIONS_FILE,
        plots_dir=TRANSFORMER_MULTIVAR_PLOTS_DIR,
        models_to_evaluate={"transformer": None},
        model_type=TRANSFORMER_MODEL,
        # selected_features_file=SELECTED_FEATURES_FILE
    )

# Read and explore the combined data and the training and test datasets
if False:
    data_dict = read_pharma_data(PHARMA_DATA_FILE, remove_nulls=True)

    # Data from spectrometer 1
    spectral_features_1 = pd.concat(
        [data_dict[KEY_CALIBRATE_1], data_dict[KEY_VALIDATE_1], data_dict[KEY_TEST_1]],
        axis=0,
    )
    # Data from spectrometer 2
    spectral_features_2 = pd.concat(
        [data_dict[KEY_CALIBRATE_2], data_dict[KEY_VALIDATE_2], data_dict[KEY_TEST_2]],
        axis=0,
    )
    target = pd.concat(
        [data_dict[KEY_CALIBRATE_Y], data_dict[KEY_VALIDATE_Y], data_dict[KEY_TEST_Y]],
        axis=0,
    )

    spectral_data_analysis(spectral_features_1, COMBINED_PHARMA_PLOTS_ROOT)
    env_data_exploration(target, COMBINED_PHARMA_PLOTS_ROOT)

    X_train = pd.concat(
        [data_dict[KEY_CALIBRATE_1], data_dict[KEY_VALIDATE_1]],
        axis=0,
    )
    y_train = pd.concat([data_dict[KEY_CALIBRATE_Y], data_dict[KEY_VALIDATE_Y]], axis=0)

    spectral_data_analysis(X_train, TRAIN_PHARMA_PLOTS_ROOT)
    env_data_exploration(y_train, TRAIN_PHARMA_PLOTS_ROOT)
    spectral_data_analysis(data_dict[KEY_TEST_1], TEST_PHARMA_PLOTS_ROOT)
    env_data_exploration(data_dict[KEY_TEST_Y], TEST_PHARMA_PLOTS_ROOT)

# Train classic multivariate models on pharma data set
if False:
    data_dict = read_pharma_data(PHARMA_DATA_FILE, remove_nulls=True)
    
    X_train = pd.concat(
        [data_dict[KEY_CALIBRATE_1], data_dict[KEY_VALIDATE_1]],
        axis=0,
    )
    y_train = pd.concat([data_dict[KEY_CALIBRATE_Y], data_dict[KEY_VALIDATE_Y]], axis=0)

    # Since 70% of the data is in test set by default, we swap training and test sets
    train_multivariate_models(
        pd.DataFrame(), # Empty DF
        data_dict[KEY_TEST_1],
        data_dict[KEY_TEST_Y],
        models_dir=CLASSIC_PHARMA_MULTIVAR_ML_DIR,
    )

    evaluate_models(
        [pd.DataFrame(), data_dict[KEY_TEST_1], data_dict[KEY_TEST_Y]],
        [pd.DataFrame(), X_train, y_train],
        models_dir=CLASSIC_PHARMA_MULTIVAR_ML_DIR,
        metrics_output_file=CLASSIC_PHARMA_MULTIVAR_ML_METRICS_FILE,
        predictions_output_file=CLASSIC_PHARMA_MULTIVAR_ML_PREDICTIONS_FILE,
        plots_dir=CLASSIC_PHARMA_MULTIVAR_ML_PLOTS_DIR,
    )

