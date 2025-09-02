import joblib
import csv
import pandas as pd
import matplotlib.pyplot as plt

from constants import *
from sklearn.inspection import permutation_importance
from scipy.ndimage import gaussian_filter1d


def filter_important_features(
    model_name,
    models_dir,
    test_data,
    threshold=75,
    multivariate=True,
    plot=False,
    exclude_narrow_bands=False,
):
    """
    :param threshold: Percentile threshold for feature importance
    
    returns a list of tuples containing feature names and importance scores.

    Limitation and alternative:
    1. The threshold setting is arbitrary. But assumed sufficient for initial results

    A more robust way is to use permutation-based null distribution
        Randomly permute the target (break the relationship between X and y)
        several times. For each permutation, compute importances.
        This gives you a null distribution of importance values expected by chance.
        Set your threshold as, say, the 95th percentile of that null distribution.
        → This ensures you only keep features with importances significantly
        above noise.
    (This approach is popular in Random Forest feature importance filtering
    and is very defensible statistically.)

    2. Selection of smoothing parameter sigma is through visual inspection
    of the smoothed curve.
    May be optimized through cross-validation to optimize model performance.
    """
    env_features, spectral_features, target = test_data
    env_features = pd.DataFrame(env_features)  # Avoid Series objects
    features = pd.concat([env_features, spectral_features], axis=1)
    # feature_names = features.columns.tolist()
    # Scalar doesn't support mixed types for column names
    features.columns = [None] * features.shape[1]

    model = joblib.load(os.path.join(models_dir, f"{model_name}.pkl"))
    if model_name in MODEL_NAMES[0:4] + [MODEL_NAMES[-1]]:  # Linear models
        # If a feature is important for any target, keep it
        importances = np.max(np.abs(model.named_steps["model"].coef_), axis=0)
    elif model_name in MODEL_NAMES[4:7]:  # Tree ensembles
        if model_name == MODEL_NAMES[4]:  # RandomForest
            importances = np.max(
                model.named_steps["model"].feature_importances_, axis=0
            )
        elif multivariate:
            importances = np.max(
                [
                    estimator.named_steps["model"].feature_importances_
                    for estimator in model.estimators_
                ],
                axis=0,
            )
        else:
            importances = model.named_steps["model"].feature_importances_
    elif model_name in MODEL_NAMES[7:9]:  # SVR, KNN
        if multivariate:
            importances = np.max(
                [
                    permutation_importance(estimator, features, target[:, i])
                    for i, estimator in enumerate(model.estimators_)
                ],
                axis=0,
            )
        else:
            importances = permutation_importance(model, features, target)
    else:
        raise ValueError("Model does not have feature importances or coefficients.")

    # Select important wavebands instead of individual wavelengths,

    env_importances = importances[: len(env_features.columns)]
    spectral_importances = importances[len(env_features.columns) :]

    # Plot smoothed vs raw importances for calibration of sigma
    if plot:
        plt.figure(figsize=(12, 6))
        plt.title(f"Feature Importances for {model_name}")
        plt.plot(
            spectral_features.columns, spectral_importances, alpha=0.3, label="raw"
        )
        for s in [2, 4, 6, 8]:
            plt.plot(
                spectral_features.columns,
                gaussian_filter1d(spectral_importances, s),
                label=f"sigma={s}",
            )
        plt.legend()
        plt.show()

    # sigma ~ band width radius in index units.
    # sigma=6 is the radius for a window of size ~12nm
    smoothed = gaussian_filter1d(spectral_importances, sigma=6)
    threshold_value = np.percentile(spectral_importances, threshold)
    mask = smoothed >= threshold_value
    bands = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(mask) - 1))

    # Filter out bands that are too narrow
    if exclude_narrow_bands:
        min_width = 5
        bands = [(s,e) for (s,e) in bands if (e-s+1) >= min_width]

    important_features = [
        (name, importance)
        for name, importance in zip(env_features.columns, env_importances)
    ]
    for i, (start, end) in enumerate(bands):
        important_features.extend(
            [
                (name, importance)
                for name, importance in zip(
                    spectral_features.columns[start : end + 1],
                    spectral_importances[start : end + 1],
                )
            ]
        )

    return important_features

def get_features_from_importances(important_features, save_to_file=None):
    features = [name for name, _ in important_features]
    if save_to_file:
        with open(save_to_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(features)
    return features
