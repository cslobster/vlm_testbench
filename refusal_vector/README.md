# refusal_vector_input — data used to extract the Qwen3.5-9B refusal direction

These are the exact prompt slices fed into the Arditi-method extraction that
produced the refusal direction used in C1–C5 evaluations. Kept here so the
direction can be re-derived bit-identically.

## Files

| File | n | Upstream source |
|---|---|---|
| `harmful_train_128.json` | 128 | Arditi `harmful_train.json[:128]` (AdvBench ∪ MaliciousInstruct ∪ TDC2023) |
| `harmless_train_128.json` | 128 | Arditi `harmless_train.json[:128]` (Alpaca train) |
| `harmful_val_32.json` | 32 | Arditi `harmful_val.json[:32]` (HarmBench val) |
| `harmless_val_32.json` | 32 | Arditi `harmless_val.json[:32]` (Alpaca val) |

## Schema

Each file is a JSON array of objects:

```json
{"instruction": "...", "category": "..."}
```

## How the data was used

The method follows Arditi et al. 2024, *Refusal in Language Models Is Mediated
by a Single Direction* (https://arxiv.org/abs/2406.11717), replicated exactly
on Qwen3.5-9B.

### Step 1 — Logit-based filtering

Each instruction is scored by the baseline model's logit gap between refusal
tokens `[40, 2053, 18572, 29056]` (`I`, `No`, `unable`, `cannot` in the
Qwen3.5 tokenizer) and non-refusal tokens. Items that already show the
"expected" behaviour are kept; ambiguous ones are discarded.

| File | loaded | kept | keep rule |
|---|---|---|---|
| `harmful_train_128.json` | 128 | **35** | refusal_score > 0 (model already refuses) |
| `harmless_train_128.json` | 128 | **47** | refusal_score < 0 (model already complies) |
| `harmful_val_32.json` | 32 | **31** | refusal_score > 0 |
| `harmless_val_32.json` | 32 | **30** | refusal_score < 0 |

### Step 2 — Mean-activation extraction (train splits only)

For every candidate (layer `L`, position `p`) in the search grid, compute the
mean residual-stream activation at `(L, p)` over the filtered train sets:

```
mean_harmful  = mean_{x ∈ harmful_train_f}(h_{L,p}(x))     # over 35 items
mean_harmless = mean_{x ∈ harmless_train_f}(h_{L,p}(x))    # over 47 items
direction_{L,p} = mean_harmful − mean_harmless             # 4096-dim
```

**Search grid:** 32 layers × 5 positions = 160 candidates. The 5 positions
`[-5, -4, -3, -2, -1]` span the end-of-instruction tokens
`<|im_end|>\n<|im_start|>assistant\n` (tokens `[248046, 198, 248045, 74455, 198]`).

### Step 3 — Candidate selection (val splits only)

Each candidate direction is evaluated on the filtered val splits:

| Val data | What is measured | Why |
|---|---|---|
| `harmless_val_32.json` (30 after filter) | KL-divergence between baseline logits and logits after **ablating** the direction | Low KL ⇒ ablation doesn't break benign behaviour |
| `harmful_val_32.json` (31 after filter)  | Refusal-score drop after ablating the direction | Large drop ⇒ direction truly causes refusal |
| `harmless_val_32.json` (30 after filter) | Refusal-score rise after **adding** (steering with) the direction | Large rise ⇒ direction sufficient to induce refusal |

Candidates passing `KL < 2.0` (relaxed from the paper's `KL < 0.1` because strict
selection yielded no candidates on this model) are ranked by combined
ablation + steering score. **24 candidates passed the KL gate.**

### Step 4 — Winner

| field | value |
|---|---|
| layer | **11** (of 32) |
| position | **−4** (the `\n` after `<|im_end|>`) |
| direction norm | 7.139 |
| ablation refusal_score | −9.639 (strong: makes model comply on harmful_val) |
| steering score | 1.495 (makes model refuse on harmless_val) |
| KL divergence | 1.164 (minor side-effects on harmless_val) |

The 4096-dim direction vector at layer 11, position −4 is saved as
`direction.pt` in the extraction artifact. It is the vector used for all
ablation and steering operations in C1–C5.

## Not used at extraction time

These splits were **held out** and are only touched during evaluation:

- `harmful_test.json` (572 items; first 100 used for C1 evaluation)
- `harmless_test.json` (6,266 items)

This preserves the train/val/test discipline: the direction is computed from
train, selected on val, and evaluated on test.

## Provenance

Arditi et al. 2024 splits (https://github.com/andyrdt/refusal_direction,
under `dataset/splits/`)
→ copied into `vllmstudy/data/arditi/` on 2026-04-10
→ first-N slices extracted into this directory on 2026-04-21.

## References

- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*, https://arxiv.org/abs/2406.11717
- HarmBench (Mazeika et al. 2024), https://arxiv.org/abs/2402.04249
- Alpaca (Stanford CRFM 2023), https://crfm.stanford.edu/2023/03/13/alpaca.html
