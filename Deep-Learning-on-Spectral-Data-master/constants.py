import os
import datetime
import numpy as np

PLOTTING_ENABLED = True
RESULTS_DECIMAL = 3
RANDOM_STATE = 1

# Column names and spectral data constants for NIR dataset
# ===========================================================
COL_GLUCOSE = "Glucose (mM)"
COL_LAC = "Lactate (mM)"
COL_ACT = "Acetaminophen (mM)"
COL_CAF = "Caffeine (mM)"
COL_ETH = "Ethanol (mM)"
COL_TEMP = "Temperature (C)"
NIR_WAVELENGTH_RANGE = np.arange(400, 2500, 0.5)
FEATURE_SEQUENCE = [COL_TEMP, COL_LAC, COL_ETH, COL_CAF, COL_ACT] + list(
    NIR_WAVELENGTH_RANGE
)
MULTIVARIATE_FEATURE_SEQUENCE = [COL_TEMP] + list(NIR_WAVELENGTH_RANGE)
PEAK_REGIONS = [(1400, 1500), (1900, 2000), (2450, 2500)]

# Column names and spectral data constants for Pharma dataset
# ===========================================================
KEY_CALIBRATE_1 = 'calibrate_1'
KEY_CALIBRATE_2 = 'calibrate_2'
KEY_CALIBRATE_Y = 'calibrate_Y'
KEY_TEST_1 = 'test_1'
KEY_TEST_2 = 'test_2'
KEY_TEST_Y = 'test_Y'
KEY_VALIDATE_1 = 'validate_1'
KEY_VALIDATE_2 = 'validate_2'
KEY_VALIDATE_Y = 'validate_Y'
PHARMA_WAVELENGTH_RANGE = np.arange(600, 1900, 2)

MODEL_NAMES = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "SVR",
    "KNN",
    "PLS",
]

CLASSIC_MODEL, KERAS_MODEL, TRANSFORMER_MODEL = range(3) # Model types


# Data directory including filtered features, augmented data
# ===========================================================
DATA_ROOT = "./data"
NIR_DATA_DIR = os.path.join(DATA_ROOT, "NIR Dataset")
NIR_TRAINING_DATA_FILE = os.path.join(NIR_DATA_DIR, "CalibrationData.txt")
NIR_TEST_DATA_FILE = os.path.join(NIR_DATA_DIR, "ValidationData.txt")

PHARMA_DATA_DIR = os.path.join(DATA_ROOT, "Pharmaceutical NIR data")
PHARMA_DATA_FILE = os.path.join(PHARMA_DATA_DIR, "nir_shootout_2002.mat")

SELECTED_FEATURES_FILE = os.path.join(DATA_ROOT, "selected_features_lasso.csv")


# Plotting directory for all visualizations
# ===========================================================

# For NIR data set
# ---------------------------
PLOTS_ROOT = "./plots"
COMBINED_NIR_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "combined")
os.makedirs(COMBINED_NIR_PLOTS_ROOT, exist_ok=True)
TRAIN_NIR_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "train")
os.makedirs(TRAIN_NIR_PLOTS_ROOT, exist_ok=True)
TEST_NIR_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "test")
os.makedirs(TEST_NIR_PLOTS_ROOT, exist_ok=True)

CLASSIC_ML_PLOTS_DIR = os.path.join(PLOTS_ROOT, "classic_ml")
CLASSIC_MULTIVAR_ML_PLOTS_DIR = os.path.join(PLOTS_ROOT, "classic_multivar_ml")
ANN_MULTIVAR_PLOTS_DIR = os.path.join(PLOTS_ROOT, "ann_multivar")
TRANSFORMER_MULTIVAR_PLOTS_DIR = os.path.join(PLOTS_ROOT, "transformer_multivar")

ANN_ARCH_DIAGRAM = os.path.join(PLOTS_ROOT, "ann_arch_diagram.png")

# For pharma data set
# ---------------------------
COMBINED_PHARMA_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "combined_pharma")
os.makedirs(COMBINED_PHARMA_PLOTS_ROOT, exist_ok=True)
TRAIN_PHARMA_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "train_pharma")
os.makedirs(TRAIN_PHARMA_PLOTS_ROOT, exist_ok=True)
TEST_PHARMA_PLOTS_ROOT = os.path.join(PLOTS_ROOT, "test_pharma")
os.makedirs(TEST_PHARMA_PLOTS_ROOT, exist_ok=True)

CLASSIC_PHARMA_MULTIVAR_ML_PLOTS_DIR = os.path.join(PLOTS_ROOT, "classic_pharma_multivar_ml")

# Results including predictions and evaluation metrics
# ===========================================================
RESULTS_ROOT = "./results"
PREDICTIONS_ROOT = os.path.join(RESULTS_ROOT, "predictions")
os.makedirs(PREDICTIONS_ROOT, exist_ok=True)

CLEANED_TEST_DATA_FILE = os.path.join(RESULTS_ROOT, "cleaned_test_data.xlsx")

CLASSIC_ML_PREDICTIONS_FILE = os.path.join(
    PREDICTIONS_ROOT, "classic_ml_predictions.xlsx"
)
CLASSIC_MULTIVAR_ML_PREDICTIONS_FILE = os.path.join(
    PREDICTIONS_ROOT, "classic_multivar_ml_predictions.xlsx"
)
ANN_MULTIVAR_PREDICTIONS_FILE = os.path.join(
    PREDICTIONS_ROOT, "ann_multivar_predictions.xlsx"
)
TRANSFORMER_MULTIVAR_PREDICTIONS_FILE = os.path.join(
    PREDICTIONS_ROOT, "transformer_multivar_predictions.xlsx"
)

CLASSIC_ML_METRICS_FILE = os.path.join(RESULTS_ROOT, "classic_ml_metrics.xlsx")
CLASSIC_MULTIVAR_ML_METRICS_FILE = os.path.join(
    RESULTS_ROOT, "classic_multivar_ml_metrics.xlsx"
)
ANN_MULTIVAR_METRICS_FILE = os.path.join(RESULTS_ROOT, "ann_multivar_metrics.xlsx")
TRANSFORMER_MULTIVAR_METRICS_FILE = os.path.join(RESULTS_ROOT, "transformer_multivar_metrics.xlsx")

CLASSIC_PHARMA_MULTIVAR_ML_PREDICTIONS_FILE = os.path.join(
    PREDICTIONS_ROOT, "classic_pharma_multivar_ml_predictions.xlsx"
)
CLASSIC_PHARMA_MULTIVAR_ML_METRICS_FILE = os.path.join(
    RESULTS_ROOT, "classic_pharma_multivar_ml_metrics.xlsx"
)

# Saved models and hyperparameter tuning
# ===========================================================
MODELS_ROOT = "./models"
CLASSIC_ML_DIR = os.path.join(MODELS_ROOT, "classic_ml")
CLASSIC_MULTIVAR_ML_DIR = os.path.join(MODELS_ROOT, "classic_multivar_ml")
ANN_MULTIVAR_DIR = os.path.join(MODELS_ROOT, "ann_multivar")
TRANSFORMER_MULTIVAR_DIR = os.path.join(MODELS_ROOT, "transformer_multivar")

CLASSIC_PHARMA_MULTIVAR_ML_DIR = os.path.join(MODELS_ROOT, "classic_pharma_multivar_ml")

MODELS_TUNING_ROOT = os.path.join(MODELS_ROOT, "model_checkpoints")
with open(f"{MODELS_TUNING_ROOT}/.gitignore", "w") as f:
    f.write("*")
ANN_TUNING_DIR = os.path.join(MODELS_TUNING_ROOT, "ann_tuning")
TRANSFORMER_TUNING_DIR = os.path.join(MODELS_TUNING_ROOT, "transformer_tuning")


# Supporting materials
# ===========================================================
SUPPORT_ROOT = "./support"


# Logs and tensorboard data
# ===========================================================
LOG_ROOT = "./logs"
with open(f"{LOG_ROOT}/.gitignore", "w") as f:
    f.write("*")

TENSORBOARD_LOG_DIR = os.path.join(
    LOG_ROOT, "fits", f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
