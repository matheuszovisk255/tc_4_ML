import os
import io
import json
import time
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from joblib import load as joblib_load
from keras.models import load_model

from numpy.lib.stride_tricks import sliding_window_view

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sklearn.preprocessing import MinMaxScaler

from .schemas import (
    PredictRequest,
    PredictResponse,
    YFinancePredictRequest,
    YFinancePredictResponse,
    BacktestResponse,
)

# ============================================================
# MLflow: import opcional (não quebra a API se mlflow não existir)
# ============================================================
try:
    import mlflow
except Exception:
    mlflow = None

# ============================================================
# Prometheus Metrics
# ============================================================
REQUEST_COUNT = Counter("api_requests_total", "Total de requests na API", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latência por endpoint (s)", ["endpoint"])
INFER_LATENCY = Histogram("model_inference_latency_seconds", "Latência de inferência do modelo (s)")

# ============================================================
# Cache simples do yfinance em memória (pra acelerar)
# ============================================================
_YF_CACHE: Dict[Any, Any] = {}
_YF_TTL = int(os.getenv("YF_CACHE_TTL", str(60 * 30)))  # default 30 min

# ============================================================
# Performance do backtest/predict
# ============================================================
MAX_BACKTEST_POINTS = int(os.getenv("MAX_BACKTEST_POINTS", "600"))
PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "1024"))

# ============================================================
# Helpers básicos
# ============================================================
def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

USE_DYNAMIC_SCALER = _to_bool(os.getenv("USE_DYNAMIC_SCALER", "0"))

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

def _validate_date(s: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(s, format="%Y-%m-%d", errors="raise")
    except Exception:
        raise HTTPException(status_code=422, detail=f"Data inválida '{s}'. Use YYYY-MM-DD.")

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)
    eps = 1e-9
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

def _dynamic_scaler_fit_on_train(close_values: np.ndarray, split: int) -> MinMaxScaler:
    # Fit no treino (até split) pra evitar vazamento
    if split <= 1:
        raise HTTPException(status_code=422, detail="Dados insuficientes para ajustar scaler dinâmico.")
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(close_values[:split].reshape(-1, 1))
    return sc

# ============================================================
# MLflow config (tudo via env)
# ============================================================
def _mlflow_on() -> bool:
    # liga só se mlflow existe + MLFLOW_ENABLED=1
    return (mlflow is not None) and _to_bool(os.getenv("MLFLOW_ENABLED", "0"))

def _mlflow_cfg() -> Dict[str, Any]:
    return {
        "enabled": _mlflow_on(),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        "experiment": (os.getenv("MLFLOW_EXPERIMENT", "lstm-fastapi") or "lstm-fastapi").strip(),
        "log_backtest": _to_bool(os.getenv("MLFLOW_LOG_BACKTEST", "1")),
        "log_predict": _to_bool(os.getenv("MLFLOW_LOG_PREDICT", "0")),
        "log_predict_yf": _to_bool(os.getenv("MLFLOW_LOG_PREDICT_YF", "0")),
        "artifact_max_rows": int(os.getenv("MLFLOW_ARTIFACT_MAX_ROWS", "400")),
    }

def _mlflow_setup() -> None:
    # Configura tracking uri + experiment uma vez no startup
    if not _mlflow_on():
        return

    cfg = _mlflow_cfg()
    uri = cfg["tracking_uri"]
    exp = cfg["experiment"]

    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)

    print(f"[MLflow] enabled=1 tracking_uri='{mlflow.get_tracking_uri()}' experiment='{exp}'")

def _mlflow_log_csv(artifact_name: str, df: pd.DataFrame) -> None:
    # Loga um CSV como artifact (capando tamanho)
    cfg = _mlflow_cfg()
    max_rows = int(cfg["artifact_max_rows"])

    out = df.copy()
    if max_rows > 0 and len(out) > max_rows:
        out = out.tail(max_rows)

    mlflow.log_text(out.to_csv(index=False), artifact_file=artifact_name)

# ============================================================
# Carrega .env
# ============================================================
load_dotenv()

MODEL_PATH = _get_env("MODEL_PATH", "models/lstm_model.keras")
SCALER_PATH = _get_env("SCALER_PATH", "models/scaler.pkl")
METADATA_PATH = _get_env("METADATA_PATH", "models/metadata.json")

# ============================================================
# App FastAPI
# ============================================================
app = FastAPI(
    title="Stock LSTM Prediction API (com yfinance)",
    version="1.3.1",
    description="API para inferência de um modelo LSTM (séries temporais) e busca de histórico via yfinance.",
)

# artefatos carregados na memória
model = None
scaler = None
metadata = None
window_size = None

@app.on_event("startup")
def startup():
    global model, scaler, metadata, window_size

    # MLflow setup (se habilitado)
    _mlflow_setup()

    # metadata e window_size
    metadata = _read_json(METADATA_PATH) or {}
    window_size = int(metadata.get("window_size", 0)) if metadata.get("window_size") else 0

    # valida caminhos do modelo/scaler
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH não encontrado: {MODEL_PATH}.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"SCALER_PATH não encontrado: {SCALER_PATH}.")

    # carrega modelo e scaler
    model = load_model(MODEL_PATH)
    scaler = joblib_load(SCALER_PATH)

    # tenta inferir window_size do modelo
    try:
        inferred = int(model.input_shape[1])
        if window_size <= 0 or window_size != inferred:
            window_size = inferred
    except Exception:
        if window_size <= 0:
            window_size = 60

    print(f"[API] model_loaded=1 scaler_loaded=1 window_size={window_size}")

@app.get("/health")
def health():
    # endpoint simples pra checar config e estado
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "scaler_mode": "dynamic" if USE_DYNAMIC_SCALER else "saved",
        "use_dynamic_scaler": USE_DYNAMIC_SCALER,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "metadata_path": METADATA_PATH,
        "window_size": window_size,
        "yf_cache_ttl": _YF_TTL,
        "max_backtest_points": MAX_BACKTEST_POINTS,
        "predict_batch_size": PREDICT_BATCH_SIZE,
        "mlflow": _mlflow_cfg(),
    }

@app.get("/mlflow/test")
def mlflow_test():
    # endpoint de teste pra você ver um run aparecendo SEM depender do streamlit
    if not _mlflow_on():
        return {"ok": False, "detail": "MLFLOW_ENABLED=0 ou mlflow não instalado."}

    try:
        with mlflow.start_run(run_name=f"manual_test_{int(time.time())}"):
            mlflow.set_tag("source", "mlflow_test_endpoint")
            mlflow.log_metric("test_metric", 1.0)
        return {"ok": True, "detail": "Run criado. Veja no MLflow UI."}
    except Exception as e:
        return {"ok": False, "detail": str(e)}

# ============================================================
# Core: previsão iterativa
# ============================================================
def _predict_iterative_from_history(history: np.ndarray, horizon: int, sc) -> np.ndarray:
    if history.ndim != 1:
        history = history.reshape(-1)

    # normaliza histórico
    hist_norm = sc.transform(history.reshape(-1, 1)).reshape(-1)

    if hist_norm.shape[0] < window_size:
        raise ValueError("Histórico menor que window_size")

    # janela inicial
    window = hist_norm[-window_size:].reshape(1, window_size, 1)

    preds_norm: List[float] = []
    for _ in range(horizon):
        t0 = time.perf_counter()
        yhat = model.predict(window, verbose=0)
        INFER_LATENCY.observe(time.perf_counter() - t0)

        next_val = float(yhat.reshape(-1)[0])
        preds_norm.append(next_val)

        # desliza a janela e adiciona a previsão
        w = window.reshape(window_size)
        w = np.append(w[1:], next_val)
        window = w.reshape(1, window_size, 1)

    # desnormaliza
    preds = sc.inverse_transform(np.array(preds_norm).reshape(-1, 1)).reshape(-1)
    return preds

def _next_business_dates(last_date: pd.Timestamp, n: int) -> List[str]:
    rng = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n)
    return [d.strftime("%Y-%m-%d") for d in rng]

# ============================================================
# yfinance fetch + cache
# ============================================================
def _fetch_close_series(ticker: str, start_date: str, end_date: str, interval: str, auto_adjust: bool):
    start_ts = _validate_date(start_date)
    end_ts = _validate_date(end_date)
    if end_ts < start_ts:
        raise HTTPException(status_code=422, detail="end_date precisa ser >= start_date.")

    key = (ticker, start_date, end_date, interval, bool(auto_adjust))
    now = time.time()

    cached = _YF_CACHE.get(key)
    if cached is not None:
        ts, close_values, dates, last_obs_date = cached
        if now - ts < _YF_TTL:
            return close_values, dates, last_obs_date

    df = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Sem dados para ticker='{ticker}' no período {start_date}..{end_date}.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise HTTPException(status_code=500, detail="Não encontrei coluna 'Close' no retorno do yfinance.")
        close = df["Close"]
        if hasattr(close, "columns") and len(close.columns) > 0:
            close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise HTTPException(status_code=500, detail="Não encontrei coluna 'Close' no retorno do yfinance.")
        close = df["Close"]

    close = close.dropna()
    if close.empty:
        raise HTTPException(status_code=404, detail="Série de Close vazia após remover NaNs.")

    idx = pd.to_datetime(close.index)
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass

    dates = idx.strftime("%Y-%m-%d").tolist()
    last_obs_date = pd.to_datetime(idx[-1])
    close_values = close.values.astype(float)

    _YF_CACHE[key] = (now, close_values, dates, last_obs_date)
    return close_values, dates, last_obs_date

# ============================================================
# /predict (histórico enviado pelo usuário)
# ============================================================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    endpoint = "/predict"
    t0 = time.perf_counter()

    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        hist = np.array(req.history, dtype=float)
        if hist.size < window_size:
            raise HTTPException(status_code=422, detail=f"history precisa ter pelo menos {window_size} pontos (recebido: {hist.size}).")

        # scaler: dinâmico (fit no próprio histórico) ou salvo
        if USE_DYNAMIC_SCALER:
            sc = MinMaxScaler(feature_range=(0, 1))
            sc.fit(hist.reshape(-1, 1))
        else:
            sc = scaler

        preds = _predict_iterative_from_history(hist, req.horizon, sc)

        # MLflow: log opcional
        if _mlflow_on() and _mlflow_cfg()["log_predict"]:
            try:
                with mlflow.start_run(run_name=f"predict_{int(time.time())}"):
                    mlflow.set_tag("endpoint", endpoint)
                    mlflow.set_tag("scaler_mode", "dynamic" if USE_DYNAMIC_SCALER else "saved")
                    mlflow.log_param("window_size", int(window_size))
                    mlflow.log_param("horizon", int(req.horizon))
                    mlflow.log_param("n_history", int(hist.size))
                    mlflow.log_metric("request_latency_s", float(time.perf_counter() - t0))
            except Exception as e:
                print(f"[MLflow] FAIL predict log: {e}")

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()

        return PredictResponse(
            predictions=[float(x) for x in preds.tolist()],
            window_size=int(window_size),
            horizon=int(req.horizon),
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            metadata=metadata or None,
        )

    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)

# ============================================================
# /predict/yfinance (API busca histórico e prevê)
# ============================================================
@app.post("/predict/yfinance", response_model=YFinancePredictResponse)
def predict_yfinance(req: YFinancePredictRequest):
    endpoint = "/predict/yfinance"
    t0 = time.perf_counter()

    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        close_values, _, last_obs_date = _fetch_close_series(
            req.ticker, req.start_date, req.end_date, req.interval, req.auto_adjust
        )

        if close_values.size < window_size:
            raise HTTPException(status_code=422, detail=f"Período retornou {close_values.size} pontos, mas window_size={window_size}.")

        # scaler dinâmico (fit só no treino) ou salvo
        if USE_DYNAMIC_SCALER:
            split = int(close_values.size * 0.8)
            sc = _dynamic_scaler_fit_on_train(close_values, split)
        else:
            sc = scaler

        preds = _predict_iterative_from_history(close_values, req.horizon, sc)
        predicted_dates = _next_business_dates(last_obs_date, req.horizon)

        # MLflow: log opcional
        if _mlflow_on() and _mlflow_cfg()["log_predict_yf"]:
            try:
                with mlflow.start_run(run_name=f"predict_yf_{req.ticker}_{int(time.time())}"):
                    mlflow.set_tag("endpoint", endpoint)
                    mlflow.set_tag("ticker", req.ticker)
                    mlflow.set_tag("scaler_mode", "dynamic" if USE_DYNAMIC_SCALER else "saved")
                    mlflow.log_param("start_date", req.start_date)
                    mlflow.log_param("end_date", req.end_date)
                    mlflow.log_param("interval", req.interval)
                    mlflow.log_param("auto_adjust", bool(req.auto_adjust))
                    mlflow.log_param("window_size", int(window_size))
                    mlflow.log_param("horizon", int(req.horizon))
                    mlflow.log_param("n_obs", int(close_values.size))
                    mlflow.log_metric("request_latency_s", float(time.perf_counter() - t0))

                    df_pred = pd.DataFrame({"date": predicted_dates, "prediction": [float(x) for x in preds.tolist()]})
                    _mlflow_log_csv("predict_yfinance.csv", df_pred)
            except Exception as e:
                print(f"[MLflow] FAIL predict_yfinance log: {e}")

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()

        return YFinancePredictResponse(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            last_observation_date=last_obs_date.strftime("%Y-%m-%d"),
            predicted_dates=predicted_dates,
            n_observations_used=int(min(close_values.size, window_size)),
            predictions=[float(x) for x in preds.tolist()],
            window_size=int(window_size),
            horizon=int(req.horizon),
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            metadata=metadata or None,
        )

    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)

# ============================================================
# /backtest/yfinance (lado a lado real vs previsto no teste)
# ============================================================
@app.post("/backtest/yfinance", response_model=BacktestResponse)
def backtest_yfinance(req: YFinancePredictRequest):
    endpoint = "/backtest/yfinance"
    t0 = time.perf_counter()

    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Modelo/scaler ainda não carregados.")

        close_values, dates, _ = _fetch_close_series(
            req.ticker, req.start_date, req.end_date, req.interval, req.auto_adjust
        )

        n = close_values.size
        if n <= window_size + 5:
            raise HTTPException(status_code=422, detail="Poucos dados para backtest. Aumente o período.")

        split = int(n * 0.8)
        if split <= window_size:
            raise HTTPException(status_code=422, detail="Poucos dados: split <= window_size. Aumente o período.")

        # scaler dinâmico: fit no treino; senão usa o scaler salvo do treino original
        if USE_DYNAMIC_SCALER:
            sc = _dynamic_scaler_fit_on_train(close_values, split)
        else:
            sc = scaler

        norm = sc.transform(close_values.reshape(-1, 1)).reshape(-1)

        X_all = sliding_window_view(norm, window_size)[:-1]
        y_all = close_values[window_size:]
        d_all = dates[window_size:]

        start_pos = split - window_size
        X_test = X_all[start_pos:]
        y_true = y_all[start_pos:]
        d_test = d_all[start_pos:]

        if MAX_BACKTEST_POINTS > 0 and X_test.shape[0] > MAX_BACKTEST_POINTS:
            X_test = X_test[-MAX_BACKTEST_POINTS:]
            y_true = y_true[-MAX_BACKTEST_POINTS:]
            d_test = d_test[-MAX_BACKTEST_POINTS:]

        X_test = X_test.reshape(-1, window_size, 1).astype(np.float32)

        t_inf = time.perf_counter()
        pred_norm = model.predict(X_test, verbose=0, batch_size=PREDICT_BATCH_SIZE).reshape(-1, 1)
        INFER_LATENCY.observe(time.perf_counter() - t_inf)

        y_pred = sc.inverse_transform(pred_norm).reshape(-1)
        metrics = _compute_metrics(y_true, y_pred)

        # MLflow: esse é o MAIS útil (métricas do backtest)
        if _mlflow_on() and _mlflow_cfg()["log_backtest"]:
            try:
                with mlflow.start_run(run_name=f"backtest_{req.ticker}_{int(time.time())}"):
                    mlflow.set_tag("endpoint", endpoint)
                    mlflow.set_tag("ticker", req.ticker)
                    mlflow.set_tag("scaler_mode", "dynamic" if USE_DYNAMIC_SCALER else "saved")

                    mlflow.log_param("start_date", req.start_date)
                    mlflow.log_param("end_date", req.end_date)
                    mlflow.log_param("interval", req.interval)
                    mlflow.log_param("auto_adjust", bool(req.auto_adjust))
                    mlflow.log_param("window_size", int(window_size))
                    mlflow.log_param("split_index", int(split))
                    mlflow.log_param("n_obs", int(n))

                    mlflow.log_metric("mse", float(metrics["mse"]))
                    mlflow.log_metric("rmse", float(metrics["rmse"]))
                    mlflow.log_metric("mae", float(metrics["mae"]))
                    mlflow.log_metric("mape", float(metrics["mape"]))
                    mlflow.log_metric("infer_latency_s", float(time.perf_counter() - t_inf))
                    mlflow.log_metric("request_latency_s", float(time.perf_counter() - t0))

                    df_bt = pd.DataFrame({"date": list(d_test), "y_true": y_true.tolist(), "y_pred": y_pred.tolist()})
                    _mlflow_log_csv("backtest.csv", df_bt)
            except Exception as e:
                print(f"[MLflow] FAIL backtest log: {e}")

        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()

        return BacktestResponse(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            auto_adjust=req.auto_adjust,
            window_size=int(window_size),
            split_index=int(split),
            dates=list(d_test),
            y_true=[float(x) for x in y_true.tolist()],
            y_pred=[float(x) for x in y_pred.tolist()],
            metrics=metrics,
        )

    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)

# ============================================================
# Prometheus scrape
# ============================================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
