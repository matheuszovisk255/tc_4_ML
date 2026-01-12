import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv

load_dotenv()

def create_sequences(series_scaled: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_scaled)):
        X.append(series_scaled[i - lookback:i, 0])
        y.append(series_scaled[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(lookback: int, units1: int, units2: int, dropout: float, lr: float):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(units1, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units2, return_sequences=False),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="DIS")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units1", type=int, default=64)
    parser.add_argument("--units2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    exp_train = os.getenv("MLFLOW_EXPERIMENT_TRAIN", "stock-lstm")
    mlflow.set_experiment(exp_train)

    # Data
    df = yf.download(args.symbol, start=args.start, end=args.end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    data = df[["Date", "Close"]].dropna().reset_index(drop=True)

    close_values = data["Close"].values.reshape(-1, 1)

    # Split temporal
    train_ratio = 0.8
    train_size = int(len(close_values) * train_ratio)

    train = close_values[:train_size]
    test = close_values[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Sequences
    LOOKBACK = args.lookback
    X_train, y_train = create_sequences(train_scaled, LOOKBACK)

    # Val temporal (20% final do treino)
    val_ratio = 0.2
    val_size = int(len(X_train) * val_ratio)
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    # Test com contexto
    test_with_context = np.concatenate([train_scaled[-LOOKBACK:], test_scaled], axis=0)
    X_test, y_test = create_sequences(test_with_context, LOOKBACK)

    # Reshape
    X_tr = X_tr.reshape(-1, LOOKBACK, 1)
    X_val = X_val.reshape(-1, LOOKBACK, 1)
    X_test = X_test.reshape(-1, LOOKBACK, 1)

    # Train
    tf.keras.backend.clear_session()
    model = build_lstm_model(LOOKBACK, args.units1, args.units2, args.dropout, args.lr)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    with mlflow.start_run(run_name=f"{args.symbol}_LSTM"):
        mlflow.log_params({
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "lookback": args.lookback,
            "epochs": args.epochs,
            "batch": args.batch,
            "units1": args.units1,
            "units2": args.units2,
            "dropout": args.dropout,
            "lr": args.lr,
        })

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch,
            shuffle=False,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Curvas por época
        for ep in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", float(history.history["loss"][ep]), step=ep)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][ep]), step=ep)
            mlflow.log_metric("train_mae", float(history.history["mae"][ep]), step=ep)
            mlflow.log_metric("val_mae", float(history.history["val_mae"][ep]), step=ep)

        # Avaliação em escala real
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_real = scaler.inverse_transform(y_pred_scaled)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100

        mlflow.log_metric("test_mae", float(mae))
        mlflow.log_metric("test_rmse", float(rmse))
        mlflow.log_metric("test_mape", float(mape))

        # Log model no MLflow
        mlflow.tensorflow.log_model(model, artifact_path="model")

        # Salvar artefatos locais para a API
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        model.save(artifacts_dir / "model.keras")
        joblib.dump(scaler, artifacts_dir / "scaler.pkl")
        meta = {
            "symbol": args.symbol,
            "lookback": LOOKBACK,
            "train_end_date": str(data["Date"].iloc[train_size - 1].date()),
            "test_start_date": str(data["Date"].iloc[train_size].date()),
        }
        (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("\n✅ Treino concluído")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print(f"✅ Artefatos salvos em: {artifacts_dir.resolve()}")

if __name__ == "__main__":
    main()
