# Gold Advisor Bot

Web app for the CM3020 final project *A Financial Advisor Bot for Gold* (Project Idea 4.2).
It gives a Buy / Hold / Sell call for gold futures (GC=F) for the next trading day, explains it
in plain English, and shows an honest out-of-sample track record.

**Student project. Not financial advice.**

## How it works
- `advisor_core.py`: features, the Attention-GRU (exported from TensorFlow to NumPy), leakage-safe
  rolling Buy/Sell cut-offs, backtest, and an optional Gemini explanation layer with a rule-based fallback.
- `streamlit_app.py`: the web page.
- `test_app.py`: 19 automated tests (run `pip install pytest tensorflow && pytest -v`).
- `model_weights.npz`, `model_config.json`: the trained model (seed 4); `attention_gru.keras`: original Keras model.
- `sample_gold.csv`: the training-period dataset with features.

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
