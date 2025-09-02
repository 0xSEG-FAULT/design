import torch
import math
import joblib
import json
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from constants import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import savgol_filter

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR + "_transformer")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SpectralPreprocessor(nn.Module):
    def __init__(self, input_size, reduced_size=512):
        super().__init__()
        self.input_size = input_size
        self.reduced_size = reduced_size

        # Feature reduction via learned linear projection
        self.feature_reducer = nn.Linear(input_size, reduced_size)
        self.batch_norm = nn.BatchNorm1d(reduced_size)
        self.dropout = nn.Dropout(0.2)

        # Initialize with small weights to prevent saturation
        nn.init.normal_(self.feature_reducer.weight, std=0.01)
        # nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.feature_reducer.bias)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.feature_reducer(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class SpectraTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        reduced_size,
        input_channels=1,
        d_model=32,
        patch_size=16,
        stride=8,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        num_targets=5,
        dropout=0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.reduced_size = reduced_size
        self.input_channels = input_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_targets = num_targets
        self.dropout = dropout

        self.preprocessor = SpectralPreprocessor(input_size, reduced_size)
        self.num_patches = (reduced_size - patch_size) // stride + 1

        # Patch embedding via Conv1D
        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )

        # Use smaller initialization for limited data
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 10)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simpler regression head with strong regularization
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_targets),
        )

        # Careful initialization for regression head
        for module in self.reg_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.preprocessor(x)  # [batch_size, reduced_size]
        x = x.unsqueeze(1)  # [batch_size, 1, num_wavelengths] for Conv1d
        x = self.patch_embed(x)  # [batch_size, d_model, num_patches]
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        x = self.transformer(x)  # [batch_size, num_patches, d_model]
        x = torch.mean(x, dim=1)  # average pooling
        out = self.reg_head(x)  # [batch_size, num_targets]
        return out


class EarlyStopping:
    def __init__(
        self, patience=25, min_delta=1e-5, path=f"{MODELS_ROOT}/transformer.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # save best weights
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def apply_spectral_preprocessing(features, n_components=None, apply_pca=True):
    features = (
        features.fillna(features.mean()) if hasattr(features, "fillna") else features
    )
    # Savitzky-Golay smoothing (simple approximation)
    if hasattr(features, "values"):
        features_smooth = np.apply_along_axis(
            lambda x: savgol_filter(x, window_length=5, polyorder=2),
            axis=1,
            arr=features.values,
        )
        features = pd.DataFrame(features_smooth, index=features.index)

    # Optional PCA
    pca = None
    if apply_pca:
        # Keep enough components to explain 95% of variance, but cap at reasonable number
        n_components = min(
            n_components or 200, features.shape[0] - 10, features.shape[1]
        )
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        features_pca = pca.fit_transform(features)
        print(
            f"PCA reduced features from {features.shape[1]} to {features_pca.shape[1]}"
        )
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        features = pd.DataFrame(features_pca, index=features.index)

    return features, pca


def prepare_data_for_transformer(
    features, target, models_dir, batch_size, apply_spectral_prep=True
):
    if apply_spectral_prep:
        features, pca = apply_spectral_preprocessing(features, apply_pca=False)
        if pca is not None:
            joblib.dump(pca, f"{models_dir}/pca_transformer.pkl")

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

    target_scaler = StandardScaler()
    if isinstance(y_train, pd.DataFrame) and len(y_train.columns) > 1:
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
    else:
        # Scalar doesn't accept 1D array
        y_train = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_val = target_scaler.fit(y_val.to_numpy().reshape(-1, 1))
    joblib.dump(target_scaler, f"{models_dir}/target_scaler.pkl")

    X_train = torch.from_numpy(X_train).float().to(DEVICE)
    y_train = torch.from_numpy(y_train).float().to(DEVICE)
    X_val = torch.from_numpy(X_val).float().to(DEVICE)
    y_val = torch.from_numpy(y_val).float().to(DEVICE)

    trainset = TensorDataset(X_train, y_train)
    valset = TensorDataset(X_val, y_val)
    dataloader_train = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    dataloader_val = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val


def train_transformer(features, target, models_dir, max_epochs=200, batch_size=16):
    def train(dataloader, model, loss_fn, optimizer, clip_grad_norm=0.5):
        num_batches = len(dataloader)
        model.train()
        training_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()  # Zero out the gradients to build fresh ones for the next iteration
            pred = model(X)
            loss = loss_fn(pred, y)

            # L2 regularization
            l2_reg = torch.tensor(0.0, device=DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg

            training_loss += loss.item()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        return training_loss / num_batches

    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():  # Disable gradient calculation for inference
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        # Unweighted average of MSE per batch
        # Since last batch size may differ, this does not represent true MSE for the epoch
        # However, it is sufficient for monitoring
        return test_loss / num_batches

    os.makedirs(models_dir, exist_ok=True)
    model_file = f"{models_dir}/transformer.pt"

    input_size = len(features.columns)
    num_targets = len(target.columns) if isinstance(target, pd.DataFrame) else 1

    model = SpectraTransformer(
        input_size=input_size,
        reduced_size=min(512, input_size // 2),
        d_model=32,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        num_targets=num_targets,
        dropout=0.3,
    ).to(DEVICE)

    loss_fn = nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    early_stopping = EarlyStopping(patience=30, min_delta=1e-6, path=model_file)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-6
    )
    # Warm-up scheduler for first few epochs
    # warmup_epochs = 5
    # warmup_scheduler = optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    # )

    dataloader_train, dataloader_val = prepare_data_for_transformer(
        features, target, models_dir, batch_size
    )

    print(f"\nTraining Transformer...")
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        training_loss = train(dataloader_train, model, loss_fn, optimizer)
        test_loss = test(dataloader_val, model, loss_fn)
        print(f"Train Loss: {training_loss:.4f} | Val Loss: {test_loss:.4f}\n")

        writer.add_scalar("Loss/train", training_loss, epoch)
        writer.add_scalar("Loss/val", test_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        early_stopping(test_loss, model)  # Saves the best model
        if early_stopping.early_stop:
            print(f"Stopping at epoch {epoch}")
            break

        # Skipping warmup since we will have few gradient updates on small datasets
        # if epoch < warmup_epochs:
        #     warmup_scheduler.step()
        # else:
        #     lr_scheduler.step(test_loss)
        lr_scheduler.step(test_loss)

    print(f"Saved model state to {model_file}")
    results_to_save = {
        "input_size": input_size,
        "reduced_size": model.reduced_size,
        "batch_size": batch_size,
        "input_channels": model.input_channels,
        "d_model": model.d_model,
        "patch_size": model.patch_size,
        "stride": model.stride,
        "num_layers": model.num_layers,
        "nhead": model.nhead,
        "dim_feedforward": model.dim_feedforward,
        "num_targets": model.num_targets,
        "dropout": model.dropout,
    }
    with open(f"{models_dir}/best_tuned_parameters.json", "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"Tuned parameters saved to {models_dir}/best_tuned_parameters.json")
