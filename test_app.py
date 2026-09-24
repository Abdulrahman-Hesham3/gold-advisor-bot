"""
Automated tests for the Gold Advisor Bot (run with: python -m pytest -v test_app.py).

They check the properties the report relies on: the deployed model gives the same
answers as the trained TensorFlow model, the app predicts exactly as the training
pipeline does, never uses future data, only offers out-of-sample days, evaluates its
own advice correctly, stays honest, keeps working when the live data feed or the
language model is unavailable, and the web page itself renders and responds.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import advisor_core as core

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def eng():
    """Engine built from the saved data only, so results are deterministic."""
    real_fetch = core.fetch_live_prices
    core.fetch_live_prices = lambda after: (None, "disabled for tests")
    try:
        engine = core.Engine()
    finally:
        core.fetch_live_prices = real_fetch
    return engine


def _raw_windows(eng, dates):
    X = eng.scale_features(eng.feat)
    pos = [eng.feat.index.get_loc(d) for d in dates]
    return np.stack([X[p - eng.lookback:p] for p in pos])


# ---------------------------------------------------------------- model + data
def test_numpy_model_matches_tensorflow(eng):
    """The exported NumPy network gives the same forecasts as the trained Keras model."""
    tf = pytest.importorskip("tensorflow")
    path = Path(core.ART_DIR) / "attention_gru.keras"
    if not path.exists():
        pytest.skip("attention_gru.keras not present")

    class AttentionLayer(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                     initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                     initializer="zeros", trainable=True)
            super().build(input_shape)

        def call(self, x):
            e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            a = tf.nn.softmax(e, axis=1)
            return tf.reduce_sum(x * a, axis=1)

    keras_model = tf.keras.models.load_model(path, custom_objects={"AttentionLayer": AttentionLayer},
                                             compile=False)
    dates = eng.table.index[::50]
    windows = _raw_windows(eng, dates).astype("float32")
    tf_out = keras_model.predict(windows, verbose=0).ravel()
    np_out = eng.model.predict(windows)
    assert np.max(np.abs(tf_out - np_out)) < 1e-4


def test_features_match_training_pipeline(eng):
    """Indicators recomputed by the app equal the ones the model was trained on."""
    recomputed = core.add_features(eng.saved[core.OHLCV].astype(float))
    rows = eng.saved.index[-500:]
    diff = np.abs(recomputed.loc[rows, eng.cols].values - eng.saved.loc[rows, eng.cols].values)
    assert np.nanmax(diff) < 1e-6


def test_prediction_uses_the_60_days_before_the_date(eng):
    """The app's forecast for a date equals the model run on the 60 rows before it."""
    date = eng.table.index[len(eng.table) // 2]
    manual = eng.forecast(_raw_windows(eng, [date]))[0]
    assert abs(manual - eng.table.loc[date, "pred"]) < 1e-9


def test_only_out_of_sample_days_are_offered(eng):
    """No recommendation is ever shown for a day inside the training period."""
    assert eng.table.index.min() > eng.split_date


# ---------------------------------------------------------------- no look-ahead
def test_thresholds_use_only_earlier_forecasts(eng):
    """Each day's cut-offs are percentiles of the previous 250 forecasts only."""
    t = eng.table
    rng = np.random.default_rng(0)
    for i in rng.choice(np.arange(len(t)), size=20, replace=False):
        date = t.index[i]
        pos = eng.feat.index.get_loc(date)
        earlier = eng.forecast(_raw_windows(eng, eng.feat.index[pos - core.CAL_WINDOW:pos]))
        assert np.isclose(t.loc[date, "buy"], np.percentile(earlier, 67), atol=1e-9)
        assert np.isclose(t.loc[date, "sell"], np.percentile(earlier, 33), atol=1e-9)


def test_changing_the_future_does_not_change_past_calls(eng):
    """Scrambling every price after a date leaves that date's call unchanged."""
    date = eng.table.index[len(eng.table) // 2]
    before = eng.table.loc[date, ["pred", "buy", "sell", "action"]]
    saved = eng.saved.copy()
    later = saved.index > date
    for i, col in enumerate(saved.columns):
        saved.loc[later, col] = saved.loc[later, col].sample(frac=1, random_state=i).values
    real_fetch, real_load = core.fetch_live_prices, core.load_saved_dataset
    core.fetch_live_prices = lambda after: (None, "disabled")
    core.load_saved_dataset = lambda art_dir=None: saved
    try:
        scrambled = core.Engine()
    finally:
        core.fetch_live_prices, core.load_saved_dataset = real_fetch, real_load
    after = scrambled.table.loc[date, ["pred", "buy", "sell", "action"]]
    assert np.isclose(before["pred"], after["pred"])
    assert np.isclose(before["buy"], after["buy"]) and np.isclose(before["sell"], after["sell"])
    assert before["action"] == after["action"]


# ---------------------------------------------------------------- decision layer
def test_decision_layer_produces_all_three_actions():
    preds = np.linspace(-0.01, 0.01, 201)
    assert {core.decide(p, 0.001, -0.001, -0.01, 0.01)[0] for p in preds} == {"BUY", "HOLD", "SELL"}


def test_signal_strength_stays_between_0_and_100(eng):
    assert eng.table["conf"].between(0, 100).all()


def test_sell_cutoff_never_above_buy_cutoff(eng):
    assert (eng.table["sell"] <= eng.table["buy"]).all()


# ---------------------------------------------------------------- evaluation of the advice
def test_buy_and_hold_reconstructs_from_daily_returns(eng):
    r = eng.table.dropna(subset=["actual"])["actual"].values
    assert np.isclose(eng.backtest["bh_cum"][-1], np.prod(1 + r))


def test_strategy_follows_the_stated_rule(eng):
    """Buy = in gold, Sell = cash, Hold = keep position, minus 0.1% per change."""
    t = eng.table.dropna(subset=["actual"])
    pos, value, trades = 0, 1.0, 0
    for action, r in zip(t["action"], t["actual"]):
        new = 1 if action == "BUY" else 0 if action == "SELL" else pos
        if new != pos:
            trades += 1
            value *= 1 - core.TRADE_COST
        pos = new
        value *= 1 + pos * r
    assert eng.backtest["trades"] == trades
    assert np.isclose(eng.backtest["strat_cum"][-1], value)


# ---------------------------------------------------------------- user-facing behaviour
def test_weekend_date_snaps_to_previous_trading_day(eng):
    idx = eng.table.index
    friday = next(d for d in idx[len(idx) // 2:] if d.weekday() == 4)
    rec = eng.recommend(friday + pd.Timedelta(days=1))
    assert rec["date"] == friday and "not a trading day" in rec["note"]


def test_date_before_range_uses_first_available_day(eng):
    rec = eng.recommend(pd.Timestamp("2005-01-03"))
    assert rec["date"] == eng.table.index[0] and "earliest day" in rec["note"]


def test_card_is_honest(eng):
    html = core.card_html(eng.recommend(None), eng)
    assert "not financial advice" in html
    assert "of the time" in html              # the coin-flip track record is on the card
    assert "What a " in html and "call means" in html  # what the call means for the position
    assert "next trading day" in html              # the forecast horizon is stated
    assert "not</b> the chance of being right" in html


def test_buy_wording_never_says_hold():
    """User test fix: 'a Buy call means holding gold' was read as 'Hold'."""
    assert "hold" not in core.STRATEGY_MEANING["BUY"].lower()
    assert "hold" not in core.STRATEGY_MEANING["SELL"].lower()


def test_strength_labels():
    assert [core.strength_word(c) for c in (0, 33.9, 34, 66.9, 67, 100)] == \
        ["Weak", "Weak", "Moderate", "Moderate", "Strong", "Strong"]


# ---------------------------------------------------------------- resilience
def test_explanation_falls_back_without_api_key(eng, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    facts = core.facts_for_llm(eng.recommend(None), eng)
    text, problem = core.explain_text(facts)
    assert problem and facts["recommendation"] in text and "not financial advice" in text


def test_question_box_fails_gracefully_without_api_key(eng, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    facts = core.facts_for_llm(eng.recommend(None), eng)
    assert "isn't available" in core.answer_text("What does RSI mean?", facts)


def test_live_data_failure_falls_back_to_saved_data(eng):
    assert "Saved dataset" in eng.source


# ---------------------------------------------------------------- the web page itself
@pytest.fixture()
def page(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(core, "fetch_live_prices", lambda after: (None, "disabled for tests"))
    at = testing.AppTest.from_file(str(HERE / "streamlit_app.py"), default_timeout=120)
    return at.run()


def _page_html(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_web_page_loads_with_a_recommendation(page):
    assert not page.exception
    html = _page_html(page)
    assert any(word in html for word in (">BUY<", ">HOLD<", ">SELL<"))
    assert "not financial advice" in html


def test_web_page_updates_when_a_weekend_day_is_picked(page, eng):
    idx = eng.table.index
    friday = next(d for d in idx[len(idx) // 2:] if d.weekday() == 4)
    page.date_input(key="day").set_value((friday + pd.Timedelta(days=1)).date()).run()
    assert not page.exception
    assert "not a trading day" in _page_html(page)


# ---------------------------------------------------------------- AI layer resilience
class _FakeResponse:
    def __init__(self, status, body):
        import json as _json
        self.status_code, self._body, self.text = status, body, _json.dumps(body)

    def json(self):
        return self._body


def test_busy_ai_model_falls_back_to_the_next_one(monkeypatch):
    """A 503 'high demand' reply from the first model hands over to the lighter backup model."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(core, "_WORKING_MODEL", None)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if len(calls) <= 2:  # first model busy on the first try and on the retry
            return _FakeResponse(503, {"error": {"message": "This model is experiencing high demand."}})
        return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "Plain answer."}]}}]})

    monkeypatch.setattr(core.requests, "post", fake_post)
    text, problem = core.call_gemini("system", "question")
    assert text == "Plain answer." and problem is None
    assert "gemini-flash-latest" in calls[0] and "gemini-flash-latest" not in calls[-1]


def test_all_models_busy_gives_a_friendly_message(eng, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(core, "_WORKING_MODEL", None)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    monkeypatch.setattr(core.requests, "post",
                        lambda url, **kw: _FakeResponse(503, {"error": {"message": "high demand"}}))
    facts = core.facts_for_llm(eng.recommend(None), eng)
    text, problem = core.answer("What does RSI mean?", facts)
    assert problem.startswith("busy") and "Try again in a minute" in text
