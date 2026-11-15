# Financial sentiment: FinBERT small vs large

**Pick a model and get positive / negative / neutral on financial text, with a side-by-side small-vs-large trade-off. The 33M MiniLM is fast; the 435M DeBERTa-large is a few points more accurate at ~19x the CPU latency. Both are weak on headlines; negative headlines are not reliably found (see Limits).**

Paste a sentence (Single tab) or one headline per line (Batch tab) and read the
probabilities from whichever model is selected:

- [9mark9/finbert-minilm-sentiment](https://huggingface.co/9mark9/finbert-minilm-sentiment) — MiniLM-L12, 33.4M params, fast on CPU, **temperature-calibrated**.
- [9mark9/finbert-deberta-large-sentiment](https://huggingface.co/9mark9/finbert-deberta-large-sentiment) — DeBERTa-v3-large, 435M params, more accurate, softmax **uncalibrated** (no fitted temperature). Loaded lazily, only when first selected.

Both are fine-tuned on the same v3 data mix: Financial PhraseBank, finance tweets
and press-release headlines. The UI **abstains** below 70% top-class confidence
instead of forcing a label, and flags very short, out-of-distribution, or
uncalibrated inputs.

## What was measured

Held-out sets, never used in training or model selection. Accuracy / macro-F1.
DeBERTa numbers are for transformers 4.x (this Space pins `transformers==4.46.3`);
on transformers 5.x its tweet score drops to ~89.5%.

| Held-out set | n | MiniLM 33M | DeBERTa-large 435M |
|--------------|---|-----------|--------------------|
| PhraseBank `Sentences_AllAgree`, 15% test split | 340 | 97.94% / 0.964 | **99.41% / 0.993** |
| Finance tweets ([zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)), validation file | 2,388 | 86.43% / 0.830 | **90.16% / 0.879** |
| Press-release headlines ([Jean-Baptiste/financial_news_sentiment](https://huggingface.co/datasets/Jean-Baptiste/financial_news_sentiment)), test split | 267 | 75.28% / 0.590 | **78.28% / 0.614** |
| News headlines ([SEntFiN](https://huggingface.co/datasets/tasksource/sentfin)), test split, unanimous headlines | 933 | 75.99% / 0.763 | **79.74% / 0.803** |

The large model wins on every set, but the gap is a few points while it costs
~13x the parameters and ~19x the CPU latency — the point the toggle makes visible.

## Limits — read before quoting a number

- PhraseBank `Sentences_AllAgree` is the subset where every annotator agreed: the
  cleanest, easiest part of the benchmark. Expect lower accuracy on ambiguous text.
- The datasets define sentiment differently (tweets label bullish / bearish market
  direction), so the rows are not directly comparable.
- Both models miss most negative press-release headlines. The honest cross-domain
  drop from PhraseBank to the headline sets is real and shown above.
- **Calibration is per-model.** Only the MiniLM softmax is temperature-calibrated
  (ECE 0.284 → 0.029); the DeBERTa confidence is raw and flagged uncalibrated, so
  its 70% abstain cut-off is a rough threshold, not a calibrated probability.
- English only. Training used 96 tokens; this demo truncates at 256 tokens and
  4,000 characters. Batch tab caps at 100 lines.
- It labels the tone of one sentence. It does not predict prices and is not
  investment advice.

## Tests

`python3 -m pytest test_app.py` covers the label-mapping, temperature-softmax and
abstain logic with fake logits — no weights are downloaded.

Training details, per-class results and usage code are on each model card.
Educational from-scratch version: [github.com/shaikn6/nano-finbert](https://github.com/shaikn6/nano-finbert).
