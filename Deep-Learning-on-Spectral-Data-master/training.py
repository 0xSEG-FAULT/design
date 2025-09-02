import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import json

from constants import *
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam, Nadam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras_tuner import RandomSearch, BayesianOptimization

models_to_train = {
    MODEL_NAMES[0]: (LinearRegression(), {}),
    MODEL_NAMES[1]: (Ridge(), {"model__alpha": [1e-8, 0.1, 1, 10]}),
    MODEL_NAMES[2]: (Lasso(), {"model__alpha": [1e-8, 0.1, 1, 10]}),
    MODEL_NAMES[3]: (
        ElasticNet(),
        {"model__alpha": [1e-8, 0.1, 1, 10], "model__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    MODEL_NAMES[4]: (
        RandomForestRegressor(),
        {"model__n_estimators": [10, 100, 200], "model__max_depth": [100, 10]},
    ),
    MODEL_NAMES[5]: (
        GradientBoostingRegressor(),
        {
            "model__n_estimators": [10, 100, 200],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
    ),
    MODEL_NAMES[6]: (
        XGBRegressor(),
        {
            "model__n_estimators": [10, 100, 200],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
    ),
    MODEL_NAMES[7]: (
        SVR(),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    ),
    MODEL_NAMES[8]: (KNeighborsRegressor(), {"model__n_neighbors": [5, 10]}),
    MODEL_NAMES[9]: (PLSRegression(), {"model__n_components": [10, 20, 50]}),
}


multi_models_to_train = {
    MODEL_NAMES[0]: (LinearRegression(), {}),
    MODEL_NAMES[1]: (Ridge(), {"regressor__model__alpha": [1e-8, 0.1, 1, 10]}),
    MODEL_NAMES[2]: (Lasso(), {"regressor__model__alpha": [1e-8, 0.1, 1, 10]}),
    MODEL_NAMES[3]: (
        ElasticNet(),
        {
            "regressor__model__alpha": [1e-8, 0.1, 1, 10],
            "regressor__model__l1_ratio": [0.2, 0.5, 0.8],
        },
    ),
    MODEL_NAMES[4]: (
        RandomForestRegressor(),
        {
            "regressor__model__n_estimators": [10, 100, 200],
            "regressor__model__max_depth": [100, 10],
        },
    ),
    MODEL_NAMES[5]: (
        MultiOutputRegressor(GradientBoostingRegressor()),
        {
            "regressor__model__estimator__n_estimators": [10, 100, 200],
            "regressor__model__estimator__learning_rate": [0.05, 0.1, 0.2],
        },
    ),
    MODEL_NAMES[6]: (
        MultiOutputRegressor(XGBRegressor()),
        {
            "regressor__model__estimator__n_estimators": [10, 100, 200],
            "regressor__model__estimator__learning_rate": [0.05, 0.1, 0.2],
        },
    ),
    MODEL_NAMES[7]: (
        MultiOutputRegressor(SVR()),
        {
            "regressor__model__estimator__C": [0.1, 1, 10],
            "regressor__model__estimator__kernel": ["linear", "rbf"],
        },
    ),
    MODEL_NAMES[8]: (
        MultiOutputRegressor(KNeighborsRegressor()),
        {"regressor__model__estimator__n_neighbors": [5, 10]},
    ),
    MODEL_NAMES[9]: (PLSRegression(), {"regressor__model__n_components": [10, 20, 50]}),
}


def train_classic_models(
    env_features,
    spectral_features,
    target,
    models_dir=MODELS_ROOT,
    models_to_train=models_to_train,
):
    os.makedirs(models_dir, exist_ok=True)
    features = pd.concat([env_features, spectral_features], axis=1)
    # Scalar doesn't support mixed types for column names
    if hasattr(features, "columns"):
        features.columns = [f"feature_{i}" for i in range(features.shape[1])]

    # Train and evaluate each model
    results = {}
    for model_name, (model, param_grid) in models_to_train.items():
        print(f"\nTraining {model_name}...")
        results[model_name] = {}
        k = np.min([5, len(features)])
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        pipe = Pipeline([("scaler", feature_scaler), ("model", model)])
        pipe = TransformedTargetRegressor(regressor=pipe, transformer=target_scaler)
        grid_search = GridSearchCV(
            pipe, param_grid, scoring="neg_mean_squared_error", cv=k, verbose=2
        )
        grid_search.fit(features, target)

        # Best model
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, f"{models_dir}/{model_name}.pkl")
        results[model_name]["best_params"] = grid_search.best_params_
        results[model_name]["best_model"] = best_model
        results[model_name]["cv_results"] = grid_search.cv_results_

    with pd.ExcelWriter(f"{models_dir}/training_results.xlsx") as writer:
        for model_name, result in results.items():
            pd.DataFrame(result["cv_results"]).to_excel(
                writer, sheet_name=f"{model_name} CV Results", index=False
            )
            best_params = pd.DataFrame(result["best_params"], index=[0])
            best_params.to_excel(
                writer, sheet_name=f"{model_name} Best Params", index=False
            )
    print(f"Data saved to {models_dir}/training_results.xlsx")
    return results


def train_multivariate_models(
    env_features, spectral_features, target, models_dir=MODELS_ROOT
):
    train_classic_models(
        env_features, spectral_features, target, models_dir, multi_models_to_train
    )


class LRTensorBoard(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self, logs=None):
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        with self.writer.as_default():
            tf.summary.scalar("learning_rate", data=lr, step=epoch)


# To view results for training, use: tensorboard --logdir logs/fits
lr_logger = LRTensorBoard(TENSORBOARD_LOG_DIR+"_ANN")


def train_ann_models(features, target, models_dir):
    def build_model(hp):
        model = Sequential()

        model.add(
            Dense(
                hp.Int("units_0", 16, 256, step=16),
                kernel_regularizer=l2(hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")),
                input_shape=(len(features.columns),),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout_0", 0.0, 0.5, step=0.1)))
        model.add(Activation(hp.Choice("activation", ["relu", "elu", "swish"])))

        for i in range(hp.Int("num_layers", 1, 8)):
            model.add(
                Dense(
                    hp.Int("units_" + str(i), 16, 256, step=16),
                    kernel_regularizer=l2(
                        hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")
                    ),
                )
            )
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))
            model.add(Activation(hp.Choice("activation", ["relu", "elu", "swish"])))

        model.add(
            Dense(
                len(
                    (
                        target.to_frame() if isinstance(target, pd.Series) else target
                    ).columns
                ),
                activation="linear",
            )
        )  # Output layer

        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
        optimizer_name = hp.Choice("optimizer", ["adam", "adamw", "nadam"])
        if optimizer_name == "adam":
            opt = Adam(learning_rate=lr)
        elif optimizer_name == "adamw":
            opt = AdamW(learning_rate=lr)
        else:
            opt = Nadam(learning_rate=lr)

        model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])

        return model

    print(f"\nTraining ANNs...")
    os.makedirs(models_dir, exist_ok=True)
    # Scalar doesn't support mixed types for column names
    if hasattr(features, "columns"):
        features.columns = [f"feature_{i}" for i in range(features.shape[1])]
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    joblib.dump(feature_scaler, f"{models_dir}/feature_scaler.pkl")
    if isinstance(y_train, pd.DataFrame) and len(y_train.columns) > 1:
        # Multivariate target may need scaling
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
        joblib.dump(target_scaler, f"{models_dir}/target_scaler.pkl")
    results = []
    for batch_size in [32, 64]:
        print(f"Tuning for batch size: {batch_size}")
        tuner = BayesianOptimization(
            build_model,
            objective="val_loss",
            max_trials=20,
            executions_per_trial=1,
            directory=ANN_TUNING_DIR,
            project_name=f"ann_tuning_{batch_size}",
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            TensorBoard(
                log_dir=TENSORBOARD_LOG_DIR,
                histogram_freq=1,
                write_graph=True,
                profile_batch=0,
            ),
            lr_logger,
        ]

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=50,
            callbacks=callbacks,
            verbose=1,
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        val_loss = best_model.evaluate(X_val, y_val, verbose=0)[0]

        results.append(
            {
                "batch_size": batch_size,
                "val_loss": val_loss,
                "best_hp": best_hp,
                "model": best_model,
            }
        )

    best_result = min(results, key=lambda r: r["val_loss"])
    best_result["model"].save(f"{models_dir}/best_ANN.keras")

    results_to_save = {
        "batch_size": best_result["batch_size"],
        "best_hp": best_result["best_hp"].values,
    }
    with open(f"{models_dir}/best_tuned_parameters.json", "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"Tuned parameters saved to {models_dir}/best_tuned_parameters.json")
