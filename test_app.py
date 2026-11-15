"""Unit tests for the label-mapping / softmax / abstain core in app.py.

These exercise the pure logic with hand-built logits and a fake tokenizer/model,
so no weights are downloaded. Run: `python3 -m pytest test_app.py`.
"""

import torch

import app


_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def test_softmax_map_sums_to_one_and_reads_label_order():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    m = app._softmax_map(logits, _ID2LABEL, temperature=1.0)
    assert set(m) == {"negative", "neutral", "positive"}
    assert abs(sum(m.values()) - 1.0) < 1e-6
    # Highest logit -> highest probability, mapped to the right label name.
    assert max(m, key=m.get) == "negative"


def test_temperature_sharpens_confidence():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    hot = app._softmax_map(logits, _ID2LABEL, temperature=1.0)
    cold = app._softmax_map(logits, _ID2LABEL, temperature=0.307)
    # Lower temperature makes the top class more peaked.
    assert cold["negative"] > hot["negative"]


def test_decide_confident_label():
    probs = {"negative": 0.05, "neutral": 0.05, "positive": 0.90}
    out = app._decide(probs, n_content_tokens=8, calibrated=True)
    assert "positive" in out
    assert "Abstain" not in out
    assert "90%" in out


def test_decide_abstains_below_threshold():
    probs = {"negative": 0.34, "neutral": 0.33, "positive": 0.33}
    out = app._decide(probs, n_content_tokens=8, calibrated=True)
    assert "Abstain" in out
    # 0.34 < _OOD_MAX_PROB (0.40) -> flat-probabilities drift note.
    assert "out-of-distribution" in out


def test_decide_short_input_note():
    probs = {"negative": 0.05, "neutral": 0.05, "positive": 0.90}
    out = app._decide(probs, n_content_tokens=2, calibrated=True)
    assert "very short" in out


def test_decide_uncalibrated_note():
    probs = {"negative": 0.05, "neutral": 0.05, "positive": 0.90}
    out = app._decide(probs, n_content_tokens=8, calibrated=False)
    assert "uncalibrated" in out


class _FakeEncoded(dict):
    def __init__(self):
        super().__init__(attention_mask=torch.ones(1, 6))


class _FakeTok:
    def __call__(self, text, **kw):
        return _FakeEncoded()


class _FakeOut:
    logits = torch.tensor([[0.1, 0.2, 3.0]])


class _FakeModel:
    def __call__(self, **kw):
        return _FakeOut()


def test_classify_one_end_to_end():
    prob_map, decision = app._classify_one(
        "Revenue jumped on strong demand",
        _FakeTok(), _FakeModel(), _ID2LABEL, temperature=1.0, calibrated=True,
    )
    assert max(prob_map, key=prob_map.get) == "positive"
    assert "positive" in decision


def test_classify_one_empty_input():
    prob_map, decision = app._classify_one(
        "   ", _FakeTok(), _FakeModel(), _ID2LABEL, temperature=1.0, calibrated=True
    )
    assert prob_map is None
    assert decision == ""


def test_classify_batch_skips_blank_lines(monkeypatch):
    # Patch the loader so no real weights are fetched.
    monkeypatch.setitem(app._LOADED, app._MODELS[app._DEFAULT_MODEL]["repo"],
                        (_FakeTok(), _FakeModel(), _ID2LABEL))
    out = app.classify_batch(app._DEFAULT_MODEL, "first line\n\n  \nsecond line")
    assert "| 1 |" in out and "| 2 |" in out
    assert "| 3 |" not in out
