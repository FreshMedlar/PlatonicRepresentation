# Handoff Report: 31.91M Autoregressive Predictive Text JEPA

**Branch**: `exp/autoregressive-jepa` (branched off `main` at `e697ccd`)  
**Active Checkpoint Directory**: `checkpoints_30m_jepa/` (`816 MB`, all stages backed up individually)  
**Test Suite**: `35/35` passing (`Ran 35 tests in 11.695s - OK`)

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
/home/medlar/Projects/.venv/bin/python -m unittest discover -s tests
```

---

## 1. Current State & Architecture (`31.91M` Total Parameters)

- **Model Dimensions**: `d_model=384, num_layers=5, num_heads=12, latent_dim=384, max_steps=8, max_seq_len=64, vocab_size=4000`
  - **Encoder** (`TextJEPAEncoder`, `dropout=0.0`): `10.70M` params
  - **Predictor** (`LatentPredictor`, `dropout=0.20`): `9.32M` params
  - **Decoder** (`ConditionalTextDecoder`, `num_soft_tokens=4`): `11.89M` params
- **Clean Data Cached in `checkpoints_30m_jepa/`** (zero `Recap-DataComp` bad-view data):
  - `gv_cache.pt` (`65 MB`): `32,768` clean Good-View pairs from `PAWS` (`21,829`), `MRPC` (`2,474`), `QQP` (`25,000`), filtered `TaPaCo` (`30,000` clusters), and `20%` clean domain prose self-views.
  - `win_train_cache.pt` (`219 MB`) & `win_test_cache.pt` (`9.4 MB`): `28,000` train / `1,200` test contiguous `T=8` windows across `200` PG-19 books (book-level train/test split: `5,489` train / `662` test passages), `2,000` `FineWeb-Edu` docs, and `~4,271` `WikiHow` guides.
  - `dec_cache.pt` (`40 MB`): `40,000` clean prose/book/WikiHow/TaPaCo sentences.
  - `tokenizer.json` (`226 KB`): clean `V=4000` BPE tokenizer.

### Preserved Stage Backups in `checkpoints_30m_jepa/`
- `encoder_s1_ep22.pt` (Stage 1 initial 22 epochs) and `encoder_s1.pt` (Stage 1 refined +14 epochs)
- `predictor_s2.pt` (Stage 2 best unseen retrieval checkpoint)
- `encoder_s3_ep4.pt` / `predictor_s3_ep4.pt` (Stage 3 pass 1) and `encoder_s3.pt` / `predictor_s3.pt` (Stage 3 pass 2)
- `decoder_s4_ep50.pt` (Stage 4 50 epochs CE-only) and `decoder_s4.pt` (Stage 4 +18 epochs CE + Cycle)
- `encoder.pt`, `predictor.pt`, `decoder.pt`, `ar_benchmark_report.json`, `history.json`, `ar_loss_progression.png`

---

## 2. Benchmark Assessment (`checkpoints_30m_jepa/ar_benchmark_report.json`)

| Metric | 11.58M Pre-SADL Baseline | 31.91M Final | Assessment |
|---|---|---|---|
| `Effective Rank` | `10.58 / 128` (`8.3%`) | **`181.77 / 384`** (`47.3%`; `195.19` after S1) | **Strong** — collapse eliminated |
| `f_H` (pre-projector) | `0.0529` | **`0.8401`** (`0.8742` after S1) | **Strong** — no null-space collapse |
| `erank(H)` / `erank(Z)` | `2.71` / — | **`100.78`** / **`72.22`** | **Healthy** (`mean_cos_Z = 0.0141`, `Uniformity = -3.6812`) |
| `Alignment L_align` | `0.2713` (collapsed cone) | **`0.4938`** | **Good, still improving** |
| `STS-B Spearman` $\rho$ (`1k`) | `0.1515` | **`0.4136`** (`p = 1.3e-42`) | **Strong (~2.7x baseline)** — room for `0.55–0.65+` |
| `Held-Out Window P@1` (50-way) | — | **`18.0%` (1-step) / `20.0%` (3-step)** | **9–10x chance**, held back by `p1_erank = 6.03` |
| `WikiHow P@1` / `MRR` | `3.00%` / `0.0846` | **`7.00%` / `0.1995`** (`Rollout Cos = 0.51–0.53`) | **2.3x baseline**, still limited on 1-step title-only guesses |
| `Held-Out Books P@1` / `MRR` | — | **`6.00%` / `0.1548`** (`Step1 Cos = 0.2500`) | **3x chance**, undertrained (only 7 Stage 3 epochs) |
| `Decoder CE Loss` | `3.490` | **`1.4494`** (`Cycle CosLoss = 0.1305`) | **Much cleaner**, still descending steadily |
| `Latent Cosine Recovery` | `0.914` (collapsed cone) | **`0.6694`** (`0.72–0.87` on clean probes) | **Solid**, improves further with lower CE & cycle loss |
| `Token Overlap F1` | `0.1004` | **`0.2199`** (`+2.2x` baseline) | **Grammatical**, but still paraphrases/swaps content nouns |

### Qualitative Behavior
- **What works well**: Pre-projector and post-projector geometry are genuinely isotropic (`f_H=0.84`, `mean_cos_Z=0.014`, `SIGReg=2.55`); `STS-B` is `0.4136`; the decoder generates grammatical sentences in both literary book style (`"The revolving windows were lighted to the ground, and Mrs."`, `cos=0.866`) and UI/procedural style (`"Click the icon of the column to adjust a quality."`, `cos=0.858`).
- **What is still weak**:
  1. **Predictor output rank (`p1_erank = 6.03 / 384`)** is still compressed compared to `erank(Z) = 72.22`, limiting `Books P@1` (`6%`) and causing multi-step rollout vectors to drift in fine-grained details.
  2. **Decoder exact lexical fidelity (`Token Overlap F1 = 0.22`, `CE = 1.45`)** swaps specific entities/nouns (`"Google Chrome" -> "bar tube"`, `"olive oil" -> "Windows"`).

---

## 3. Where More Training Will Directly Improve Results (Every Curve Was Still Descending)

1. **Extend Stage 4 Decoder Training + More Clean Sentences (Highest Immediate Payoff for Generation)**:
   - **Evidence**: `CE Loss` was still dropping cleanly at Epoch 18 of the cycle pass (`1.58 -> 1.53 -> 1.49 -> 1.45`). `dec_cache.pt` currently holds only `40,000` sentences (`208` batches/epoch), which is small for an `11.89M` decoder.
   - **Action**:
     - Expand `dec_cache.pt` to `150,000–300,000` clean sentences from `PG-19` (`max_books=500+`), `FineWeb-Edu`, `WikiHow`, and `PAWS/TaPaCo` (note: reusing existing `encoder.pt` & `tokenizer.json`, only regenerating `dec_cache.pt`).
     - Resume Stage 4 (`--stages 4 --resume_s4 --cycle_weight 0.5 --dec_noise_scale 0.15 --lr_s4 5e-4`) for `40–60` more epochs until **`CE Loss < 0.8–1.0`**.
     - *Optional architectural upgrade (requires Stage 4 retrain from scratch)*: increase `num_soft_tokens` from `4 -> 8` in `DecoderConfig` (`src/training/autoregressive_trainer.py`) and `vocab_size` to `8000–12000` to eliminate subword splits (`"interprescription"`, `"cordidate"`).

2. **Extend Stage 3 Joint Co-Training & Fix Predictor Rank (`p1_erank: 6.0 -> 40+`)**:
   - **Evidence**: Stage 3 only ran `4 + 3 = 7` epochs (`650` batches/epoch out of `1,166` batches) to fit the 30-minute run cap, and `AR MSE` (`1.006 -> 0.959`), `AR Cos` (`0.335 -> 0.374`), `GV MSE` (`0.229 -> 0.177`), and `SADL` (`0.047 -> 0.043`) were all still improving monotonically. Meanwhile `p1_erank` stayed at `6.03` because `ar_mse` + `ar_cos` still allow low-rank predictor outputs.
   - **Action**:
     - Train Stage 3 for `15–25` full epochs (`--max_s3_batches 0`) on `50,000+` book/prose windows (`max_books=500+`), using `--ar_mse_weight_s3 0.05 --ar_nce_weight_s3 0.65 --ar_nce_temp 0.15`.
     - Consider adding direct **`SIGReg` or `SADL` regularization on `z_preds` (per-batch)** in `compute_joint_autoregressive_step` so the predictor cannot collapse to `p1_erank = 6` while co-adapting with `E`.

3. **Extend Stage 1 Encoder on Fresh Paraphrase + Book Self-Views (`STS-B: 0.41 -> 0.55+`)**:
   - **Evidence**: Stage 1 `Inv MSE` was still dropping at the end (`0.432 -> 0.423 -> 0.415`), and `gv_cache.pt` (`32,768` pairs) only sampled a fraction of the `49,303` PAWS/MRPC/QQP + `30,000` TaPaCo + PG-19 book sentences.
   - **Action**: If retraining from Stage 1 (or refreshing `gv_cache.pt` to `65k–100k` pairs and resuming Stage 1 with `--resume_s1 --lr_s1 1.5e-4`), expect `Alignment < 0.30` and `STS-B >= 0.55`. *(Remember: modifying `encoder.pt` requires re-running Stage 2/3/4 or Stage 3+4 afterwards.)*

4. **Longer Book Context Windows (`max_steps: 8 -> 16`)**:
   - In `PG-19` books, `T=8` sentences is ~1 short paragraph. Increasing `max_steps=16` gives the causal `LatentPredictor` much richer narrative context (`8–15` sentences) for next-sentence book prediction.

---

## 4. Modified & New Files in Workspace (Uncommitted on `exp/autoregressive-jepa`)

- Modified: `configs/config.py`, `src/evaluation/benchmark.py`, `src/models/__init__.py`, `src/models/encoder.py`, `src/models/pipeline.py`, `src/models/predictor.py`, `src/models/sigreg.py`, `src/training/gpu_fast_trainer.py`, `src/training/train_encoder.py`
- Added: `src/data/autoregressive_corpus_loader.py`, `src/models/sadl.py`, `src/training/autoregressive_trainer.py`, `tests/test_autoregressive_jepa.py`, `tests/test_sadl.py`, `PHASE1_SUMMARY.md`, `NEXT_AGENT_REPORT.md`
- Recommended cleanup once final Tier A / 30M validation is complete: move `GPUMonitor` from `src/training/gpu_fast_trainer.py` into `src/training/trainer_utils.py`, update `README.md`, and merge `exp/autoregressive-jepa` into `main`.
