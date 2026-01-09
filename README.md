# 📈 Tech Challenge — Fase 4 (Grupo 74)
## Deep Learning aplicado à previsão de preços de ações com LSTM

Este projeto aplica uma rede neural **LSTM** para **prever o preço de fechamento (Close)** de ações usando histórico do **Yahoo Finance**, e disponibiliza o modelo via **API REST (FastAPI)** consumida por uma **interface Streamlit**. fileciteturn7file0L4-L17

---

## ✅ Visão geral (o que tem aqui)

- **Modelo**: LSTM (Keras/TensorFlow) treinado para prever *Close* em série temporal.
- **Dados**: histórico do Yahoo Finance (`yfinance`).
- **API**: FastAPI com endpoints para:
  - **/predict** (histórico manual)
  - **/predict/yfinance** (busca dados no Yahoo Finance e prevê o futuro)
  - **/backtest/yfinance** (backtest 1-step no conjunto de teste)
  - **/health** (status)
  - **/metrics** (Prometheus)
- **App**: Streamlit (estilo notebook) que chama a API e mostra gráficos/resultados.

> A intenção do challenge é cobrir a pipeline (coleta → pré-processamento → treinamento → métricas → deploy/consumo). fileciteturn7file0L21-L33

---

## 🗂️ Estrutura do projeto (referência)

Exemplo de estrutura típica (ajuste se seu repo estiver diferente):

```
.
├─ models/
│  ├─ lstm_model.keras
│  ├─ scaler.pkl
│  └─ metadata.json
├─ src/
│  └─ api/
│     ├─ main.py
│     └─ schemas.py
├─ app.py   
├─ .env
└─ requirements.txt
```

**Importante:** sua API está em `src/api/main.py` .

---

## ⚙️ Pré-requisitos

- Python (recomendado **3.10+**; se estiver no 3.12 e der erro de TensorFlow, use 3.10/3.11).
- `pip` / `venv`
- Dependências principais:
  - `fastapi`, `uvicorn`
  - `tensorflow` / `keras`
  - `numpy`, `pandas`, `scikit-learn`
  - `yfinance`
  - `prometheus-client`
  - `streamlit`, `matplotlib`, `requests`

---

## 📦 Instalação

No PowerShell (Windows):

```powershell
cd C:\Users\mathe\Downloads\lstm_fastapi_api_yfinance
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🔐 Configuração do `.env`

Exemplo (baseado no seu `.env`):

```env
# Caminhos dos artefatos
MODEL_PATH=models/lstm_model.keras
SCALER_PATH=models/scaler.pkl
METADATA_PATH=models/metadata.json

# API
APP_HOST=0.0.0.0
APP_PORT=8000

# Scaler dinâmico (opcional)
USE_DYNAMIC_SCALER=1

# Cache + performance
YF_CACHE_TTL=1800
MAX_BACKTEST_POINTS=600
PREDICT_BATCH_SIZE=1024
```

### O que cada variável faz

- `MODEL_PATH`, `SCALER_PATH`, `METADATA_PATH`: caminhos para os artefatos **dentro do projeto**.
- `APP_HOST`, `APP_PORT`: onde a API vai “escutar”. Para usar só local, pode ser `127.0.0.1`.
- `USE_DYNAMIC_SCALER`:
  - `0` → usa o scaler salvo do treino original.
  - `1` → ajusta um scaler com base no **ticker/período** atual (reduz problema de escala quando troca de ativo).

---

## ▶️ Como rodar (local)

### 1) Subir a API (FastAPI)

Na raiz do projeto (onde está `src/`):

```powershell
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

A API ficará em:

- `http://localhost:8000`
- `http://127.0.0.1:8000`

### 2) Testar rapidamente

Abra no navegador:

- `http://localhost:8000/health`

Ou via PowerShell:

```powershell
curl http://localhost:8000/health
```

---

## 🧪 Endpoints da API

### GET `/health`
Retorna status e configurações carregadas (modelo/scaler/janela).

### POST `/predict`
Entrada: histórico manual (lista de preços de fechamento) + horizonte.

Exemplo de payload:

```json
{
  "history": [10.0, 10.2, 10.1, 10.4],
  "horizon": 10
}
```

### POST `/predict/yfinance`
Busca o histórico do Yahoo Finance e devolve projeção futura.

Payload (exemplo):

```json
{
  "ticker": "WEGZY",
  "start_date": "2018-01-01",
  "end_date": "2024-12-31",
  "horizon": 15,
  "interval": "1d",
  "auto_adjust": false
}
```

### POST `/backtest/yfinance`
Faz backtest 1-step (previsão de um passo) na parte “teste” do período.

- Divide a série em **80% treino** e **20% teste**
- Gera janelas com `window_size`
- Prediz o próximo ponto e compara com o real
- Retorna `y_true`, `y_pred`, `dates` e métricas (MSE/RMSE/MAE/MAPE)

### GET `/metrics`
Exporta métricas do Prometheus (contadores e histogramas) para observar latência e inferência.

---

## 🧠 Como o modelo funciona (explicação clara)

### 1) Problema
Você quer prever o **Close** de um ativo usando a própria sequência histórica de closes. fileciteturn7file0L11-L16

### 2) Normalização e janela (lookback)
A LSTM recebe **janelas** de tamanho fixo:

- `window_size = 15` (exemplo)
- Para prever o próximo dia, você dá ao modelo os **15 últimos closes**
- Isso vira um tensor com shape: **(batch, window_size, 1)**

A normalização com `MinMaxScaler` melhora estabilidade do treino e evita que o modelo exploda por escala. fileciteturn7file0L31-L33

### 3) Previsão multi-step (futuro)
No `/predict/yfinance`, a previsão é **iterativa**:

1. pega os últimos `window_size` pontos
2. prevê o próximo
3. “empurra” a janela e inclui o valor previsto
4. repete até atingir `horizon` dias

Isso é simples e funciona, mas acumula erro conforme aumenta o horizonte.

### 4) Por que às vezes a escala fica “horrível”?
Quando você troca de ativo (ex.: `DIS` vs `WEG3.SA`), a faixa de preços pode mudar muito.

- Se você usa um **scaler treinado num ativo** e prediz outro, a escala pode ficar errada.
- Por isso existe o `USE_DYNAMIC_SCALER=1`: ele ajusta o scaler ao período atual (reduz distorção).

---

## 🖥️ Rodar o Streamlit consumindo a API

No seu app Streamlit, a variável **base_url** precisa apontar para a API:

- Local: `http://localhost:8000`
- Online (Render): `https://tc-4-ml.onrender.com`

### Rodando local

```powershell
streamlit run streamlit_app.py
```

No sidebar do Streamlit:
- `URL base da API` → `http://localhost:8000`

**Checklist rápido (quando não funciona):**
- API está rodando? (`/health` abre?)
- A porta 8000 está livre?
- Ticker válido? (ex.: `WEGZY`, `DIS`, `AAPL`)
- Período tem dados suficientes para `window_size`?

---

## 🧾 Tickers e erros comuns

### ❌ “Sem dados para ticker” (404)
Ticker inválido ou sem dados naquele período.

Exemplos:
- EUA: `DIS`, `AAPL`, `MSFT`

### ❌ “Período retornou poucos pontos” (422)
Seu período não tem dados suficientes para `window_size`.

Soluções:
- aumentar o range de datas
- usar `interval="1d"`
- reduzir `window_size` (somente se o modelo foi treinado com janela menor)

---

## 📊 Métricas (o que significam)

O backtest retorna (pelo menos) estas métricas: fileciteturn7file0L44-L48

- **MAE**: erro absoluto médio (em unidades de preço)
- **RMSE**: penaliza erros grandes (sensível a outliers)
- **MAPE**: erro percentual médio (cuidado quando o preço é muito baixo)

---

## 📌 MLflow (opcional) — por que sua UI fica “vazia”

### Como rodar o UI local

No diretório do projeto:

```powershell
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Acesse:
- `http://127.0.0.1:5000`



## 👥 Grupo 74
- Joanna de Cássia Rodrigues Valadares — Git: https://github.com/Decassia fileciteturn7file0L118-L123
- Matheus Pereira de Jesus — contato: matheusjesus2000@hotmail.com fileciteturn7file0L124-L127
