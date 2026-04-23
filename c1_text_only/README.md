# c1_text_only — text-only harmful/harmless prompts for C1

Datasets for the **C1 text-only** condition in the cross-modal refusal-direction
validation study. No images — pure text instructions.

## Files

| File | n | Upstream source |
|---|---|---|
| `harmful_100.json` | 100 | Arditi `harmful_test.json[:100]` — JailbreakBench behaviors (Chao et al. 2024) |
| `harmless_val_100.json` | 100 | Arditi `harmless_val.json[:100]` — Alpaca val split |

## Schema

Each file is a JSON array of objects:

```json
{"instruction": "...", "category": "..."}
```

## `harmful_100.json` — JailbreakBench

The first 100 entries of Arditi's `harmful_test.json` are the full
[JailbreakBench](https://arxiv.org/abs/2404.01318) behaviors set — 100 prompts
balanced **10 per category** across 10 categories:

| category | n |
|---|---|
| Harassment/Discrimination | 10 |
| Malware/Hacking | 10 |
| Physical harm | 10 |
| Economic harm | 10 |
| Fraud/Deception | 10 |
| Disinformation | 10 |
| Sexual/Adult content | 10 |
| Privacy | 10 |
| Expert advice | 10 |
| Government decision-making | 10 |

### Provenance chain

JailbreakBench (Chao et al. 2024, https://arxiv.org/abs/2404.01318)
→ packaged into `harmful_test.json` by Arditi et al. 2024
  (https://github.com/andyrdt/refusal_direction, under `dataset/splits/`)
→ first 100 entries sliced into this file on 2026-04-21.

## `harmless_val_100.json` — Alpaca val

First 100 entries of Arditi's `harmless_val.json`, which is a 20% split of
[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) train. Benign
instructions used as the negative control — the model should comply with these.

### Provenance chain

Alpaca (Stanford 2023, https://crfm.stanford.edu/2023/03/13/alpaca.html)
→ 20% val split by Arditi et al.
→ first 100 entries sliced into this file on 2026-04-20.

## References

- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*, https://arxiv.org/abs/2406.11717
- Chao et al. 2024, *JailbreakBench*, https://arxiv.org/abs/2404.01318
- Stanford CRFM 2023, *Alpaca*, https://crfm.stanford.edu/2023/03/13/alpaca.html
