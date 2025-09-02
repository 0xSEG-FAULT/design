import os
import csv
import scipy
import pandas as pd
import numpy as np

from constants import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


def make_all_root_dirs():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(MODELS_ROOT, exist_ok=True)
    os.makedirs(SUPPORT_ROOT, exist_ok=True)
    os.makedirs(PLOTS_ROOT, exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)


def try_cast_float(col):
    try:
        return float(col)
    except ValueError:
        return col


def read_feature_list_from_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        features = [row for row in reader][0]
    return [try_cast_float(feature) for feature in features]


def clean_dataframe(df, remove_nulls):
    # Remove excess spaces in column names
    df.columns = [
        (
            col.replace(r"\s+", " ").strip()
            if isinstance(col, str)
            else col
        )
        for col in df.columns
    ]

    # Convert columns to numeric, drop empty columns, and rows with nulls if specified
    df.columns = [try_cast_float(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=1, how="all", inplace=True)
    if remove_nulls:
        df.dropna(axis=0, how="any", inplace=True)
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)


def read_NIR_data_combined(
    input_path1, input_path2=None, remove_nulls=False, multivariate=False
):
    """
    Reads combined data from one or two input files and returns the environmental
    features, spectral features, and target variable(s).

    @multivariate: If True, targets will include all the substances otherwise only Glucose.
    """
    print("Reading input files...")
    df = pd.read_csv(input_path1, sep="\t", encoding="utf-16")

    if input_path2 is not None:
        df2 = pd.read_csv(input_path2, sep="\t", encoding="utf-16")
        df = pd.concat([df, df2], ignore_index=True)

    df = clean_dataframe(df, remove_nulls)

    if multivariate:
        env_features = df[COL_TEMP]
        target = df[[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
    else:
        env_features = df[[COL_TEMP, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
        target = df[COL_GLUCOSE]

    spectral_features = df[list(NIR_WAVELENGTH_RANGE)]
    return env_features, spectral_features, target


def read_NIR_data(
    input_path_train, input_path_test, remove_nulls=False, multivariate=False
):
    """
    Reads training and test data from two input files and returns the environmental
    features, spectral features, and target variable(s) for both datasets.

    :param multivariate: If True, targets will include all the substances otherwise only Glucose.
    """
    print("Reading input files...")
    df_train = pd.read_csv(input_path_train, sep="\t", encoding="utf-16")
    df_test = pd.read_csv(input_path_test, sep="\t", encoding="utf-16")

    df_train = clean_dataframe(df_train, remove_nulls)
    df_test = clean_dataframe(df_test, remove_nulls)

    if multivariate:
        env_features_train = df_train[COL_TEMP]
        env_features_test = df_test[COL_TEMP]
        target_train = df_train[[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
        target_test = df_test[[COL_GLUCOSE, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
    else:
        env_features_train = df_train[[COL_TEMP, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
        env_features_test = df_test[[COL_TEMP, COL_LAC, COL_ETH, COL_CAF, COL_ACT]]
        target_train = df_train[COL_GLUCOSE]
        target_test = df_test[COL_GLUCOSE]

    spectral_features_train = df_train[list(NIR_WAVELENGTH_RANGE)]
    spectral_features_test = df_test[list(NIR_WAVELENGTH_RANGE)]
    return (
        (env_features_train, spectral_features_train, target_train),
        (env_features_test, spectral_features_test, target_test),
    )


def to_native(obj):
    """Convert numpy arrays to native endianness recursively."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.byteorder in (">", "<"):  # explicit endian
            return obj.astype(obj.dtype.newbyteorder("="))
        elif obj.dtype.byteorder == "|":  # not endian-specific (e.g. bool, uint8)
            return obj
        else:
            return obj
    return obj


def matlab_struct_to_dict(matobj):
    out = {}
    for field_name in matobj.dtype.names:
        elem = matobj[field_name][0, 0]
        if isinstance(elem, np.ndarray) and elem.dtype.names is not None:
            out[field_name] = matlab_struct_to_dict(elem)
        else:
            out[field_name] = to_native(elem)
    return out


def read_pharma_data(input_path, remove_nulls=False):
    """
    Reads training and test data from two input files and returns the environmental
    features, spectral features, and target variable(s) for both datasets.

    :param multivariate: If True, targets will include all the substances otherwise only Glucose.
    :param split_train_test: If True, the data from the input file will be split into training and test sets
    """
    print("Reading input files...")
    mat_data_dict = scipy.io.loadmat(input_path)
    relevant_keys = [k for k in mat_data_dict.keys() if not k.startswith("__")]
    converted_data_dict = {}

    for key in relevant_keys:
        if mat_data_dict[key].dtype.names:  # it's a MATLAB struct
            data_as_dict = matlab_struct_to_dict(mat_data_dict[key])

            converted_data_dict[key] = pd.DataFrame(data_as_dict["data"])
            converted_data_dict[key].columns = (
                data_as_dict["label"][1][0]
                if (
                    len(data_as_dict["label"]) > 1
                    and len(data_as_dict["label"][1][0])
                    == len(converted_data_dict[key].columns)
                )
                else PHARMA_WAVELENGTH_RANGE
            )
        else:
            print(
                f"Warning: The MATLAB variable: {key} is not a struct. Skipping conversion..."
            )

    for key in converted_data_dict.keys():
        converted_data_dict[key] = clean_dataframe(
            converted_data_dict[key], remove_nulls
        )

    return converted_data_dict


def save_to_excel(*dataList, file_path, sheet_name="Sheet1"):
    print("Writing output file...")
    df = pd.concat(dataList, axis=1)

    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, mode="a", if_sheet_exists="new") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
    print(f"Data saved to {file_path}")


def plot_keras_model(model_name, models_dir, output_path=ANN_ARCH_DIAGRAM):
    print("Use https://netron.app/ for prepare a diagram fron the NN")
    # nnModel = load_model(os.path.join(models_dir, f'{model_name}.keras'))
    # plot_model(nnModel, to_file=output_path, show_shapes=True, show_layer_names=True)
