"""
=============================================================================
  LSTM-Based Energy Consumption Forecasting Pipeline
  Dataset : final_training_dataset.csv
  Target  : energy_kwh.1 (hourly kWh consumption)
  Author  : Senior ML Engineer — Time-Series Forecasting
=============================================================================

Architecture Summary
--------------------
  Input  → Feature Engineering → MinMax Scaling → Sliding-Window Sequences
         → Stacked Bi-LSTM + Dropout + BatchNorm → Dense Head → Prediction
         → Inverse Transform → MAE / RMSE / R² Evaluation → Visualisations
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – safe in any environment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, LayerNormalization, Multiply, Activation,
    Flatten, Permute, RepeatVector, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# ─────────────────────────────────────────────────────────────────────────────
#  0 ▷  GLOBAL HYPERPARAMETERS  (all in one place for easy tuning)
# ─────────────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).resolve().parents[1]   # experiments/01_campus_substation

CONFIG = {
    # --- Data ---
    "csv_path"       : str(EXP_DIR / "data" / "processed" / "final_training_dataset.csv"),
    "timestamp_col"  : "timestamp",
    "target_col"     : "energy_kwh.1",

    # --- Sequence ---
    "n_steps"        : 24,      # look-back window  (24 h = 1 day)
    "train_ratio"    : 0.80,    # chronological train / test split

    # --- Architecture ---
    "lstm_units"     : [128, 64],   # units per LSTM layer (stacked)
    "dense_units"    : [64, 32],    # units per Dense layer
    "dropout_rate"   : 0.20,
    "use_bidirectional": True,      # wrap LSTM with Bidirectional
    "use_attention"  : True,        # Bahdanau-style self-attention

    # --- Training ---
    "batch_size"     : 32,
    "epochs"         : 150,
    "learning_rate"  : 1e-3,
    "val_split"      : 0.15,        # fraction of train used for validation

    # --- Callbacks ---
    "es_patience"    : 15,
    "lr_patience"    : 7,
    "lr_factor"      : 0.5,
    "lr_min"         : 1e-6,

    # --- Outputs ---
    "model_path"     : str(EXP_DIR / "models" / "best_lstm_model.keras"),
    "plot_path"      : str(EXP_DIR / "outputs" / "figures" / "lstm_results.png"),
}

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
#  1 ▷  DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Load CSV, parse timestamps, sort chronologically, interpolate
    missing values, and return cleaned DataFrame + feature column list.

    Why interpolation over drop?
    → Time-series data has temporal correlation.  A linear interpolation
      between adjacent hourly readings is far more faithful than deleting
      rows and creating artificial time gaps.
    """
    print("=" * 65)
    print("  STEP 1 — LOADING & PREPROCESSING")
    print("=" * 65)

    df = pd.read_csv(cfg["csv_path"])
    print(f"  Raw shape            : {df.shape}")

    # ── Parse & sort timestamp ────────────────────────────────────────────────
    df[cfg["timestamp_col"]] = pd.to_datetime(df[cfg["timestamp_col"]])
    df.sort_values(cfg["timestamp_col"], inplace=True)
    df.set_index(cfg["timestamp_col"], inplace=True)
    df.index.freq = pd.infer_freq(df.index)   # infer hourly freq if regular

    print(f"  Date range           : {df.index.min()}  →  {df.index.max()}")
    print(f"  Rows                 : {len(df):,}")

    # ── Handle missing values ─────────────────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    df.interpolate(method="time", inplace=True)       # linear in time
    df.bfill(inplace=True)                            # edge cases at start
    df.ffill(inplace=True)                            # edge cases at end
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values fixed : {missing_before} → {missing_after}")

    # ── Identify feature columns ──────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != cfg["target_col"]]
    print(f"  Feature columns ({len(feature_cols)}) : {feature_cols}")
    print(f"  Target column        : {cfg['target_col']}")

    return df, feature_cols


# =============================================================================
#  2 ▷  FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Add time-aware and statistical features to improve model accuracy.

    Why these help:
    ▸ Lag features   – give the model explicit memory of past values
                       beyond what the LSTM window alone provides.
    ▸ Rolling stats  – smooth out noise; capture short-term trend & variance.
    ▸ Calendar feats – energy has strong daily/weekly periodicity; encoding
                       hour and day-of-week as cyclic (sin/cos) avoids
                       discontinuities at midnight / end-of-week.
    ▸ Interaction    – heat index (temp × humidity) is a known driver of
                       HVAC load, and thus energy consumption.
    """
    print("\n" + "=" * 65)
    print("  STEP 2 — FEATURE ENGINEERING")
    print("=" * 65)

    df = df.copy()

    # ── Lag features (last 1, 2, 3, 6, 12, 24 h) ─────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"energy_lag_{lag}"] = df[target_col].shift(lag)

    # ── Rolling statistics (3 h, 6 h, 24 h windows) ──────────────────────────
    for w in [3, 6, 24]:
        df[f"energy_roll_mean_{w}"] = (
            df[target_col].shift(1).rolling(w).mean()
        )
        df[f"energy_roll_std_{w}"] = (
            df[target_col].shift(1).rolling(w).std()
        )

    # ── Cyclic calendar encoding ──────────────────────────────────────────────
    hour        = df.index.hour
    dow         = df.index.dayofweek
    month       = df.index.month

    df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow  / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow  / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Temperature × Humidity interaction (heat-index proxy) ─────────────────
    if "out_temp" in df.columns and "hum_out" in df.columns:
        df["heat_index"] = df["out_temp"] * df["hum_out"] / 100.0

    # ── Drop rows with NaN introduced by lags/rolling ─────────────────────────
    df.dropna(inplace=True)

    n_engineered = len(df.columns)
    print(f"  Total features after engineering : {n_engineered}")
    print(f"  Rows remaining after lag drop    : {len(df):,}")

    return df


# =============================================================================
#  3 ▷  SCALING
# =============================================================================

def scale_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_end_idx: int,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Fit scalers ONLY on training data to prevent data leakage.
    Returns scaled X (features) and y (target) arrays.
    """
    print("\n" + "=" * 65)
    print("  STEP 3 — SCALING")
    print("=" * 65)

    X_raw = df[feature_cols].values
    y_raw = df[[target_col]].values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Fit only on train portion
    scaler_X.fit(X_raw[:train_end_idx])
    scaler_y.fit(y_raw[:train_end_idx])

    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw).flatten()

    print(f"  Features scaled : {X_scaled.shape[1]} columns")
    print(f"  Target range    : [{y_raw.min():.2f}, {y_raw.max():.2f}] kWh")
    print(f"  Scaler fitted on first {train_end_idx:,} rows (train only)")

    return X_scaled, y_scaled, scaler_X, scaler_y


# =============================================================================
#  4 ▷  SEQUENCE CREATION (SLIDING WINDOW)
# =============================================================================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window supervised format.

    For each time step t, the input is:
        X[t-n_steps : t]   →   shape (n_steps, n_features)
    and the label is:
        y[t]               →   next-step energy consumption

    This avoids any look-ahead bias because y[t] is only the *next*
    timestep, never inside the feature window.
    """
    Xs, ys = [], []
    for i in range(n_steps, len(X)):
        Xs.append(X[i - n_steps : i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def split_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    train_end_idx: int,
    n_steps: int,
) -> tuple:
    """
    Chronological (time-based) train / test split — no shuffling.
    The cut-point is shifted by n_steps to align sequences correctly.
    """
    print("\n" + "=" * 65)
    print("  STEP 4 — SEQUENCE CREATION & TRAIN / TEST SPLIT")
    print("=" * 65)

    split = train_end_idx - n_steps
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"  Sequence length (n_steps) : {n_steps}")
    print(f"  X_train shape             : {X_train.shape}")
    print(f"  X_test  shape             : {X_test.shape}")
    print(f"  y_train shape             : {y_train.shape}")
    print(f"  y_test  shape             : {y_test.shape}")

    return X_train, X_test, y_train, y_test


# =============================================================================
#  5 ▷  ATTENTION MECHANISM
# =============================================================================

def attention_block(lstm_out):
    """
    Bahdanau-style additive self-attention over the LSTM time steps.

    Why it helps:
    The model can learn to *weight* which past timesteps matter most
    for the current prediction — e.g., peak-load hours from yesterday
    may be more relevant than low-load hours 20 steps back.
    This makes the representation richer than simply taking the last
    hidden state.
    """
    # lstm_out : (batch, n_steps, units)
    # Score each time step
    score = Dense(1, activation="tanh")(lstm_out)      # (batch, n_steps, 1)
    score = Flatten()(score)                            # (batch, n_steps)
    attention_weights = Activation("softmax")(score)   # (batch, n_steps)

    # Weighted sum over time dimension
    attention_weights = RepeatVector(
        K.int_shape(lstm_out)[-1]
    )(attention_weights)                                # (batch, units, n_steps)
    attention_weights = Permute([2, 1])(attention_weights)  # (batch, n_steps, units)

    context = Multiply()([lstm_out, attention_weights]) # (batch, n_steps, units)
    context = Lambda(lambda x: K.sum(x, axis=1))(context)  # (batch, units)
    return context


# =============================================================================
#  6 ▷  MODEL BUILDING
# =============================================================================

def build_model(
    n_steps: int,
    n_features: int,
    cfg: dict,
) -> Model:
    """
    Stacked Bi-LSTM with optional Bahdanau attention and Dense head.

    Architecture:
        Input (n_steps, n_features)
        → BiLSTM-128  [return_sequences=True]  + BatchNorm + Dropout
        → BiLSTM-64   [return_sequences=True]  + BatchNorm + Dropout
        → Attention (weighted context vector)
        → Dense-64  (relu) + LayerNorm + Dropout
        → Dense-32  (relu)
        → Dense-1   (linear)   ← energy prediction

    Why Bidirectional?
    → The look-back window contains full historical context in both
      directions *within* the window, so a Bi-LSTM can capture
      patterns from both ends — e.g., the ramp-up at window start and
      the plateau at window end — before making the prediction.

    Why BatchNorm between LSTM layers?
    → Stabilises training by normalising pre-activations, enabling
      higher learning rates and reducing sensitivity to weight init.
    """
    print("\n" + "=" * 65)
    print("  STEP 5 — MODEL ARCHITECTURE")
    print("=" * 65)

    inputs = Input(shape=(n_steps, n_features), name="time_series_input")
    x = inputs

    lstm_units = cfg["lstm_units"]

    # ── Stacked LSTM layers ───────────────────────────────────────────────────
    for i, units in enumerate(lstm_units):
        return_seq = True   # always True so attention can see all time steps
        if cfg["use_bidirectional"]:
            x = Bidirectional(
                LSTM(units, return_sequences=return_seq,
                     activation="tanh", recurrent_activation="sigmoid"),
                name=f"bi_lstm_{i+1}"
            )(x)
        else:
            x = LSTM(
                units, return_sequences=return_seq,
                activation="tanh", recurrent_activation="sigmoid",
                name=f"lstm_{i+1}"
            )(x)

        x = BatchNormalization(name=f"bn_{i+1}")(x)
        x = Dropout(cfg["dropout_rate"], name=f"drop_lstm_{i+1}")(x)

    # ── Attention ─────────────────────────────────────────────────────────────
    if cfg["use_attention"]:
        x = attention_block(x)
        print("  Attention mechanism  : ENABLED")
    else:
        # Fall back to using the last time step
        x = Lambda(lambda t: t[:, -1, :], name="last_step")(x)

    # ── Dense head ────────────────────────────────────────────────────────────
    for j, units in enumerate(cfg["dense_units"]):
        x = Dense(units, activation="relu", name=f"dense_{j+1}")(x)
        x = LayerNormalization(name=f"ln_{j+1}")(x)
        x = Dropout(cfg["dropout_rate"] / 2, name=f"drop_dense_{j+1}")(x)

    output = Dense(1, activation="linear", name="prediction")(x)

    model = Model(inputs=inputs, outputs=output, name="BiLSTM_Attention")

    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n  Total trainable parameters : {total_params:,}")

    return model


# =============================================================================
#  7 ▷  TRAINING
# =============================================================================

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> tf.keras.callbacks.History:
    """
    Train with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.

    Callback rationale:
    ▸ EarlyStopping      – halts when val_loss stops improving; avoids
                           memorising train noise.
    ▸ ReduceLROnPlateau  – shrinks LR when progress stalls, enabling
                           the optimiser to navigate flatter minima.
    ▸ ModelCheckpoint    – saves best weights so the final eval always
                           uses the generalization-optimal checkpoint.
    """
    print("\n" + "=" * 65)
    print("  STEP 6 — TRAINING")
    print("=" * 65)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["es_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["lr_factor"],
            patience=cfg["lr_patience"],
            min_lr=cfg["lr_min"],
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=cfg["model_path"],
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=cfg["val_split"],
        callbacks=callbacks,
        shuffle=False,          # CRITICAL: maintain temporal order
        verbose=1,
    )

    print(f"\n  Training finished after {len(history.history['loss'])} epochs.")
    return history


# =============================================================================
#  8 ▷  EVALUATION
# =============================================================================

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Inverse-transform predictions back to kWh, then compute:
        MAE   – mean absolute error (same unit as target)
        RMSE  – root mean squared error (penalises large errors more)
        R²    – coefficient of determination (1.0 = perfect)
        MAPE  – mean absolute percentage error
    """
    print("\n" + "=" * 65)
    print("  STEP 7 — EVALUATION")
    print("=" * 65)

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_true = scaler_y.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    print(f"\n  ┌{'─'*40}┐")
    print(f"  │  {'Metric':<15} {'Value':>20}  │")
    print(f"  ├{'─'*40}┤")
    print(f"  │  {'MAE':<15} {mae:>18.4f}  │")
    print(f"  │  {'RMSE':<15} {rmse:>18.4f}  │")
    print(f"  │  {'R² Score':<15} {r2:>18.4f}  │")
    print(f"  │  {'MAPE (%)':<15} {mape:>18.4f}  │")
    print(f"  └{'─'*40}┘")

    # ── Grade ─────────────────────────────────────────────────────────────────
    if r2 >= 0.95:
        grade = "Excellent ✓✓"
    elif r2 >= 0.90:
        grade = "Very Good ✓"
    elif r2 >= 0.80:
        grade = "Good"
    else:
        grade = "Needs Improvement"
    print(f"\n  Model Grade : {grade}  (R² = {r2:.4f})")

    return y_true, y_pred, metrics


# =============================================================================
#  9 ▷  VISUALISATION
# =============================================================================

def plot_results(
    history: tf.keras.callbacks.History,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    save_path: str,
) -> None:
    """
    4-panel diagnostic figure:
      (a) Training & validation loss curves
      (b) Full actual vs. predicted series (test set)
      (c) Zoomed last 168 h (7 days) for visual sharpness
      (d) Residual (error) distribution
    """
    print("\n" + "=" * 65)
    print("  STEP 8 — VISUALISATION")
    print("=" * 65)

    fig = plt.figure(figsize=(20, 14), facecolor="#0D1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
    DARK, ACCENT, BLUE, RED, GREEN = (
        "#0D1117", "#161B22", "#58A6FF", "#FF7B72", "#3FB950"
    )

    ax_style = dict(facecolor=ACCENT,
                    labelcolor="white", titlecolor="white")

    def _style(ax, title):
        ax.set_facecolor(ACCENT)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")

    # ── (a) Loss curves ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "Training vs Validation Loss (MSE)")
    ax1.plot(history.history["loss"],     color=BLUE,  lw=1.8, label="Train Loss")
    ax1.plot(history.history["val_loss"], color=RED,   lw=1.8, linestyle="--",
             label="Val Loss")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("MSE Loss", color="white")
    ax1.legend(facecolor=DARK, edgecolor="#30363D", labelcolor="white")
    ax1.grid(alpha=0.15)

    # ── (b) Full actual vs predicted ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "Actual vs Predicted — Full Test Set")
    ax2.plot(y_true,  color=BLUE,  lw=1.2, alpha=0.85, label="Actual")
    ax2.plot(y_pred,  color=RED,   lw=1.2, alpha=0.85, linestyle="--",
             label="Predicted")
    ax2.set_xlabel("Time Step (hours)", color="white")
    ax2.set_ylabel("Energy (kWh)", color="white")
    ax2.legend(facecolor=DARK, edgecolor="#30363D", labelcolor="white")
    ax2.grid(alpha=0.15)

    # ── (c) Zoomed 7-day window ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "Zoomed View — Last 168 Hours (7 Days)")
    N = min(168, len(y_true))
    ax3.plot(y_true[-N:],  color=BLUE, lw=1.5, label="Actual")
    ax3.plot(y_pred[-N:],  color=RED,  lw=1.5, linestyle="--", label="Predicted")
    ax3.fill_between(range(N), y_true[-N:], y_pred[-N:],
                     alpha=0.12, color=GREEN)
    ax3.set_xlabel("Time Step (hours)", color="white")
    ax3.set_ylabel("Energy (kWh)", color="white")
    ax3.legend(facecolor=DARK, edgecolor="#30363D", labelcolor="white")
    ax3.grid(alpha=0.15)

    # ── (d) Residuals ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "Prediction Residuals Distribution")
    residuals = y_true - y_pred
    ax4.hist(residuals, bins=50, color=BLUE, edgecolor=DARK, alpha=0.85)
    ax4.axvline(0, color=RED, lw=2, linestyle="--", label="Zero error")
    ax4.axvline(np.mean(residuals), color=GREEN, lw=1.5, linestyle="-.",
                label=f"Mean = {np.mean(residuals):.2f}")
    ax4.set_xlabel("Residual (kWh)", color="white")
    ax4.set_ylabel("Count", color="white")
    ax4.legend(facecolor=DARK, edgecolor="#30363D", labelcolor="white")
    ax4.grid(alpha=0.15)

    # ── Metric annotation ─────────────────────────────────────────────────────
    metric_str = (
        f"MAE={metrics['MAE']:.3f}  |  "
        f"RMSE={metrics['RMSE']:.3f}  |  "
        f"R²={metrics['R2']:.4f}  |  "
        f"MAPE={metrics['MAPE']:.2f}%"
    )
    fig.text(0.5, 0.02, metric_str, ha="center", fontsize=12,
             color="#8B949E",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=ACCENT,
                       edgecolor="#30363D", alpha=0.85))

    fig.suptitle("LSTM Energy Consumption Forecasting — Results Dashboard",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK)
    print(f"  Plot saved → {save_path}")


# =============================================================================
#  10 ▷  MAIN PIPELINE
# =============================================================================

def main():
    cfg = CONFIG

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    df, base_feature_cols = load_and_preprocess(cfg)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df, cfg["target_col"])

    # Update feature list after engineering (includes lag/roll/calendar cols)
    all_feature_cols = [c for c in df.columns if c != cfg["target_col"]]

    # ── 3. Train/test index (time-based, no shuffle) ──────────────────────────
    n_total      = len(df)
    train_end    = int(n_total * cfg["train_ratio"])

    # ── 4. Scale ──────────────────────────────────────────────────────────────
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(
        df, all_feature_cols, cfg["target_col"], train_end
    )

    # ── 5. Create sequences ───────────────────────────────────────────────────
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, cfg["n_steps"])

    X_train, X_test, y_train, y_test = split_sequences(
        X_seq, y_seq, train_end, cfg["n_steps"]
    )

    # ── 6. Build model ────────────────────────────────────────────────────────
    n_features = X_train.shape[2]
    model = build_model(cfg["n_steps"], n_features, cfg)

    # ── 7. Train ──────────────────────────────────────────────────────────────
    history = train_model(model, X_train, y_train, cfg)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    y_true, y_pred, metrics = evaluate_model(model, X_test, y_test, scaler_y)

    # ── 9. Visualise ──────────────────────────────────────────────────────────
    plot_results(history, y_true, y_pred, metrics, cfg["plot_path"])

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Best model saved → {cfg['model_path']}")
    print(f"  Plots saved      → {cfg['plot_path']}")
    print("=" * 65)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
