"""
Financial Sentiment — FinBERT (MiniLM 33M vs DeBERTa-large 435M).

Two fine-tunes of the same v3 data mix (Financial PhraseBank, finance tweets,
press-release headlines), picked from one dropdown so the small-vs-large
accuracy/latency trade-off is visible:

  * 9mark9/finbert-minilm-sentiment   — MiniLM-L12, 33.4M params, fast on CPU.
  * 9mark9/finbert-deberta-large-sentiment — DeBERTa-v3-large, 435M params,
    ~13x the weights and ~19x the CPU latency, a few points more accurate.

Both classify financial text as positive / negative / neutral via softmax over
config.id2label. The MiniLM softmax is temperature-calibrated; the large model's
is left raw (no fitted temperature) and its confidence is flagged uncalibrated.
Real inference, CPU. The large model is loaded lazily, only when first selected.
"""

from __future__ import annotations

import torch

# gradio and transformers are imported lazily (UI built in __main__, weights
# loaded in _load) so the pure softmax/abstain core is importable — and testable
# — with only torch installed.

# --- Shared decision knobs ---
_MAX_TOKENS = 256
_MAX_CHARS = 4000  # hard input cap to keep latency bounded and block abuse
# Below this top-class probability the UI abstains instead of forcing a label.
# At 0.70 the calibrated MiniLM keeps ~84% of inputs at ~91% accuracy.
_ABSTAIN_THRESHOLD = 0.70
# Drift heuristics — flag inputs the model is poorly placed to judge.
_MIN_CONTENT_TOKENS = 4  # very short / low-context input
_OOD_MAX_PROB = 0.40  # flat probabilities across all classes -> possible OOD

# --- Model registry ---------------------------------------------------------
# temperature: softmax temperature scaling. The MiniLM value (0.307) was fitted
#   offline by minimising NLL on a held-out half of the twitter-financial-news
#   validation split; it cut ECE from 0.284 (T=1) to 0.029. No such fit exists
#   for the large model, so it runs at T=1 and its confidence is UNCALIBRATED.
# metrics: held-out accuracy / macro-F1, measured on data never used in training
#   or model selection. transformers 4.x numbers (this Space pins 4.46.3).
_MODELS: dict[str, dict] = {
    "MiniLM 33M (fast)": {
        "repo": "9mark9/finbert-minilm-sentiment",
        "temperature": 0.307,
        "calibrated": True,
        "params": "33.4M",
        "speed": "fast on CPU (baseline)",
        "metrics": [
            ("PhraseBank test", 340, "97.94%", "0.964"),
            ("Finance tweets", 2388, "86.43%", "0.830"),
            ("Press-release headlines", 267, "75.28%", "0.590"),
            ("SEntFiN news headlines", 933, "75.99%", "0.763"),
        ],
    },
    "DeBERTa-large 435M (accurate)": {
        "repo": "9mark9/finbert-deberta-large-sentiment",
        "temperature": 1.0,
        "calibrated": False,
        "params": "435M",
        "speed": "~19x MiniLM CPU latency",
        "metrics": [
            ("PhraseBank test", 340, "99.41%", "0.993"),
            ("Finance tweets", 2388, "90.16%", "0.879"),
            ("Press-release headlines", 267, "78.28%", "0.614"),
            ("SEntFiN news headlines", 933, "79.74%", "0.803"),
        ],
    },
}
_DEFAULT_MODEL = "MiniLM 33M (fast)"

# Lazy cache: repo -> (tokenizer, model, id2label). Populated on first use so the
# 435M model is downloaded/loaded only when a user actually selects it.
_LOADED: dict[str, tuple] = {}
torch.set_grad_enabled(False)


def _load(choice: str) -> tuple:
    """Load (and cache) the tokenizer, model and lower-cased id2label map for a
    registry choice. Label order is read from the model config, never hard-coded
    — it varies by model."""
    repo = _MODELS[choice]["repo"]
    if repo not in _LOADED:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForSequenceClassification.from_pretrained(repo)
        mdl.eval()
        id2label = {i: mdl.config.id2label[i].lower() for i in mdl.config.id2label}
        _LOADED[repo] = (tok, mdl, id2label)
    return _LOADED[repo]


def _decide(
    prob_map: dict[str, float], n_content_tokens: int, calibrated: bool
) -> str:
    """Human-readable verdict: a confident label, or an explicit abstain, plus
    any input-drift note. Confidence is the top-class probability (calibrated
    only when the chosen model has a fitted temperature)."""
    label = max(prob_map, key=prob_map.get)
    prob = prob_map[label]

    notes = []
    if n_content_tokens < _MIN_CONTENT_TOKENS:
        notes.append("very short / low-context input")
    if prob < _OOD_MAX_PROB:
        notes.append("flat probabilities — possible out-of-distribution input")
    if not calibrated:
        notes.append("confidence uncalibrated for this model")
    drift = f"  \n_Note: {'; '.join(notes)}._" if notes else ""

    if prob < _ABSTAIN_THRESHOLD:
        return (
            f"**Abstain — low confidence.** Top class *{label}* at {prob:.0%}, "
            f"below the {_ABSTAIN_THRESHOLD:.0%} confidence threshold." + drift
        )
    return f"**{label}** · confidence {prob:.0%}" + drift


def _softmax_map(
    logits: torch.Tensor, id2label: dict[int, str], temperature: float
) -> dict[str, float]:
    """Temperature-scaled softmax over logits, keyed by label name."""
    probs = torch.softmax(logits / temperature, dim=-1)[0]
    return {id2label[i]: float(probs[i]) for i in range(len(id2label))}


def _classify_one(text: str, tok, mdl, id2label, temperature, calibrated):
    """Run one input through a loaded model. Returns (prob_map, decision).
    Validated and length-capped; any failure degrades gracefully."""
    if not isinstance(text, str):
        return None, ""
    text = text.strip()
    if not text:
        return None, ""
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS]
    try:
        encoded = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
        logits = mdl(**encoded).logits
        prob_map = _softmax_map(logits, id2label, temperature)
        # Content tokens exclude the two special tokens ([CLS], [SEP]).
        n_content = int(encoded["attention_mask"].sum()) - 2
        return prob_map, _decide(prob_map, n_content, calibrated)
    except Exception as exc:  # never leak a stack trace to the UI
        print(f"[classify] inference error: {exc}")
        return dict.fromkeys(id2label.values(), 0.0), "Inference error."


def classify(choice: str, text: str) -> tuple[dict[str, float] | None, str]:
    """Single-input entry point: returns ({label: probability}, decision md)."""
    cfg = _MODELS[choice]
    tok, mdl, id2label = _load(choice)
    return _classify_one(text, tok, mdl, id2label, cfg["temperature"], cfg["calibrated"])


def classify_batch(choice: str, text: str) -> str:
    """One headline per line -> a markdown table of per-row label + confidence,
    keeping the calibrated abstain. Blank lines are skipped."""
    cfg = _MODELS[choice]
    tok, mdl, id2label = _load(choice)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "_No input. Enter one headline per line._"

    rows = ["| # | Input | Label | Confidence |", "|---|-------|-------|-----------|"]
    for i, line in enumerate(lines[:100], start=1):  # cap rows to bound latency
        prob_map, _ = _classify_one(
            line, tok, mdl, id2label, cfg["temperature"], cfg["calibrated"]
        )
        shown = (line[:80] + "…") if len(line) > 80 else line
        shown = shown.replace("|", "\\|")
        if not prob_map:
            rows.append(f"| {i} | {shown} | — | — |")
            continue
        label = max(prob_map, key=prob_map.get)
        prob = prob_map[label]
        verdict = "abstain" if prob < _ABSTAIN_THRESHOLD else label
        rows.append(f"| {i} | {shown} | {verdict} | {prob:.0%} |")
    note = "" if len(lines) <= 100 else "\n\n_Showing first 100 of %d lines._" % len(lines)
    return "\n".join(rows) + note


def _metrics_md(choice: str) -> str:
    """Stated held-out metrics table for the chosen model."""
    cfg = _MODELS[choice]
    head = (
        f"**{choice}** · {cfg['params']} params · {cfg['speed']} · "
        f"{'temperature-calibrated' if cfg['calibrated'] else 'softmax uncalibrated'}\n\n"
        "Held-out accuracy / macro-F1 (never used in training or selection):\n\n"
        "| Held-out set | n | Accuracy | Macro-F1 |\n|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| {name} | {n} | {acc} | {f1} |" for name, n, acc, f1 in cfg["metrics"]
    )
    tail = (
        "\n\nIn-domain (PhraseBank `AllAgree`) is the clean, easy case; the "
        "headline sets are the honest cross-domain drop. The large model is a "
        "few points better everywhere, at ~13x params and ~19x CPU latency."
    )
    return head + body + tail


_EXAMPLES = [
    "The company posted record quarterly profit, beating all analyst estimates",
    "Shares plunged after the firm warned of mounting losses and layoffs",
    "The board will meet on Thursday to review the quarterly report",
    "Net loss widened sharply as costs spiraled out of control",
]

_INTRO = """
# 📈 Financial Sentiment — FinBERT (small vs large)

Classify financial text as positive / negative / neutral, and compare a compact
model against a large one on the **same** data mix:

- [**MiniLM 33M**](https://huggingface.co/9mark9/finbert-minilm-sentiment) — fast on CPU, temperature-calibrated.
- [**DeBERTa-large 435M**](https://huggingface.co/9mark9/finbert-deberta-large-sentiment) — a few points more accurate, ~19x the CPU latency, loaded on first use.

Probabilities are softmax over the model's label order. The MiniLM softmax is
**temperature-calibrated** (ECE 0.284 → 0.029); the large model's is raw and its
confidence is flagged **uncalibrated**. Either way the UI **abstains** below 70%
top-class confidence rather than force a label. English only; labels one
sentence's tone — not price prediction, not investment advice.

*Built from the educational from-scratch project at
[github.com/shaikn6/nano-finbert](https://github.com/shaikn6/nano-finbert).*
"""

def build_demo():
    """Construct the Gradio Blocks UI. gradio is imported here so the module
    stays importable (for tests) without gradio installed."""
    import gradio as gr

    with gr.Blocks(
        title="Financial Sentiment — FinBERT small vs large", analytics_enabled=False
    ) as demo:
        gr.Markdown(_INTRO)
        model_choice = gr.Radio(
            choices=list(_MODELS), value=_DEFAULT_MODEL, label="Model"
        )
        metrics = gr.Markdown(_metrics_md(_DEFAULT_MODEL))

        with gr.Tab("Single"):
            text_in = gr.Textbox(
                lines=3,
                max_length=_MAX_CHARS,
                label="Financial text",
                placeholder="e.g. Quarterly revenue jumped 24% on record cloud demand",
            )
            run_btn = gr.Button("Classify", variant="primary")
            probs_out = gr.Label(num_top_classes=3, label="Probabilities")
            decision_out = gr.Markdown(label="Decision")
            gr.Examples(examples=_EXAMPLES, inputs=text_in)
            run_btn.click(classify, [model_choice, text_in], [probs_out, decision_out])

        with gr.Tab("Batch"):
            batch_in = gr.Textbox(
                lines=8,
                label="One headline per line",
                placeholder="Record profit beats estimates\nShares plunge on loss warning\nBoard meets Thursday to review report",
            )
            batch_btn = gr.Button("Classify all", variant="primary")
            batch_out = gr.Markdown(label="Results")
            batch_btn.click(classify_batch, [model_choice, batch_in], batch_out)

        model_choice.change(_metrics_md, model_choice, metrics)
    return demo


if __name__ == "__main__":
    build_demo().launch()
