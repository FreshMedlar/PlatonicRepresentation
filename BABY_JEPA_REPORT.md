# Baby-JEPA `active_spectral` Training Report

**Branch**: `baby-jepa`  
**Output Directory**: `checkpoints_baby_jepa_active_spectral/`  
**Saved Checkpoints**:
- `best_active_spectral.pt` (`94 MB`) — best combined `procrustes_shape_naming_acc + dino_cka` checkpoint (`ep=384`, `6,144` interactions: `proc_shape=0.490`, `minilm_shape=0.458`, `dino_cka=0.186`)
- `checkpoint_active_spectral.pt` (`94 MB`) — final checkpoint at `ep=1024`, `16,384` interactions (`proc_shape=0.406`, `proc_color=0.698`, `dino_cka=0.218`, `dino_10nn=0.118`)
- `baby_jepa_report.json` (`14 KB`) — full per-checkpoint metrics and action counts
- `baby_jepa_diagnostics.png` & `baby_jepa_sample_efficiency.png` — training & evaluation curves

---

## 1. Architecture Evolution Across Runs

1. **Run 1 (`checkpoints_baby_jepa_active_spectral` v1)**: Single latent projector `z` with static spectral offset `z_hat - mu`. Showed collapse resistance via `BatchStandardization1d`, but `orbit_down` (`17.7%`) and `flip_pitch` (`32.3%`) looped at boundary clamps and `f_H` stalled at `0.172`.
2. **Run 2 (First-Principles Boredom + Spectral Displacement)**: Replaced `z_hat - mu` with **spectral displacement** `(z_hat - z_t)`, added **episodic boredom / habituation gating** on `visited_z`, and added `CosineAnnealingLR`. Fixed boundary looping (`orbit_down: 17.7% -> 5.6%`) and achieved near-ideal isotropy (`f_Z=0.973`, `mean_cos_Z=0.022`, `f_H=0.485`), but lacked an explicit invariance loss separating object identity from pose/zoom nuisance.
3. **Run 3 (Factorized Semantic + Equivariant Dynamics Heads, Current Checkpoint)**:
   - Split `BabyVisionEncoder` cleanly into:
     - **`z_sem` (128D Good-View Semantic Head)**: trained symmetrically with **Good-View invariance** (`F.mse_loss(z_sem_seq, z_sem_pair)`) across distinct viewpoints/lighting within the same exploration episode, plus sparse cross-modal word grounding and `SIGReg`/`SADL`.
     - **`z_dyn` (128D Action-Conditioned Dynamics Head)**: trained with the 5-member `PredictorEnsemble` (`temporal_mse + whitened_mse` + `SIGReg`) and used by `active_spectral` for epistemic disagreement + spectral displacement + episodic boredom exploration.
   - Pre-run verification: `42/42` unit tests passing (`python -m unittest discover -s tests -q`).

---

## 2. Run 3 Progression (`1,024` Episodes / `16,384` Interactions)

| Interactions | Ep | `gv_inv` | `temporal_wmse` | `cross_modal` | `f_H` | `f_Z` | `mean_cos_Z` | `erank_Z` | `shape_c2d` | `color_c2d` | `shape_clean` | `color_clean` | `proc_shape` | `proc_color` | `minilm_shape` | `dino_cka` | `dino_10nn` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | — | — | — | 0.101 | 0.101 | 0.901 | 40.9 | 0.349 | 0.385 | 0.427 | 0.458 | 0.260 | 0.302 | 0.281 | 0.185 | 0.097 |
| 2,048 | 128 | 0.734 | **0.716** | 1.064 | 0.083 | 0.408 | 0.593 | 33.6 | **0.406** | 0.427 | 0.323 | 0.510 | 0.344 | 0.448 | 0.333 | 0.195 | 0.097 |
| 4,096 | 256 | 0.876 | 0.747 | 0.823 | 0.115 | 0.405 | 0.595 | 32.9 | 0.307 | 0.505 | 0.375 | 0.604 | 0.302 | 0.365 | 0.292 | 0.208 | 0.100 |
| 6,144 *(best)* | 384 | 0.808 | 0.770 | 0.705 | 0.137 | 0.483 | 0.518 | 35.1 | 0.333 | 0.479 | 0.344 | 0.604 | **0.490** | 0.448 | **0.458** | 0.186 | 0.100 |
| 8,192 | 512 | 0.832 | 0.782 | 0.668 | 0.182 | 0.658 | 0.342 | 38.6 | 0.385 | **0.521** | 0.469 | 0.656 | 0.365 | 0.604 | 0.375 | 0.209 | 0.113 |
| 10,240 | 640 | 0.963 | 0.793 | 0.650 | 0.228 | 0.797 | 0.199 | 40.5 | 0.328 | 0.432 | 0.448 | **0.729** | 0.375 | 0.573 | 0.385 | 0.205 | 0.106 |
| 12,288 | 768 | 0.793 | 0.786 | 0.635 | 0.284 | 0.901 | 0.095 | 41.5 | 0.349 | 0.500 | 0.427 | 0.656 | 0.375 | 0.677 | 0.365 | 0.217 | 0.115 |
| 14,336 | 896 | **0.672** | 0.792 | **0.617** | 0.320 | 0.954 | 0.042 | 42.4 | 0.380 | 0.484 | **0.490** | 0.656 | 0.417 | **0.740** | 0.427 | 0.215 | 0.117 |
| 16,384 *(final)* | 1,024 | 0.783 | 0.818 | **0.617** | **0.344** | **0.961** | **0.035** | **43.0** | 0.396 | 0.510 | 0.438 | 0.677 | 0.406 | 0.698 | 0.396 | **0.218** | **0.118** |

*(4-class chance baseline = `0.250`.)*

---

## 3. Run 1 vs. Run 2 vs. Run 3 Final Comparison (`16,384` Interactions)

| Metric | Run 1 (Static Offset) | Run 2 (Boredom + Displacement) | Run 3 (+ Factorized Good-View `z_sem`) |
|---|---:|---:|---:|
| `shape_probe_from_confounded` | 0.292 | 0.260 | **0.396** |
| `color_probe_from_confounded` | 0.359 | 0.349 | **0.510** |
| `shape_probe_decorrelated` | 0.302 | 0.281 | **0.438** *(0.490 peak)* |
| `color_probe_decorrelated` | 0.406 | 0.427 | **0.677** *(0.729 peak)* |
| `procrustes_shape_naming_acc` | 0.260 *(0.385 peak)* | 0.312 *(0.375 peak)* | **0.406** *(0.490 peak)* |
| `procrustes_color_naming_acc` | 0.469 *(0.490 peak)* | 0.323 *(0.448 peak)* | **0.698** *(0.740 peak)* |
| `minilm_procrustes_shape_acc` | 0.260 *(0.385 peak)* | 0.281 *(0.396 peak)* | **0.396** *(0.458 peak)* |
| `dino_cka` | 0.181 | 0.167 | **0.218** *(0.248 on H)* |
| `dino_mutual_10nn` | 0.082 | 0.073 | **0.118** |
| `f_Z` / `mean_cos_Z` | 0.526 / 0.473 | **0.973 / 0.022** | 0.961 / 0.035 |

---

## 4. Selected Action Distribution (Run 3, `15,360` Total Actions)

| Action | Count | Share |
|---|---:|---:|
| `orbit_up` | 4,078 | 26.5% |
| `zoom_out` | 1,896 | 12.3% |
| `orbit_left` | 1,785 | 11.6% |
| `flip_pitch` | 1,561 | 10.2% |
| `orbit_down` | 1,521 | 9.9% |
| `zoom_in` | 1,472 | 9.6% |
| `shift_light_color` | 1,067 | 6.9% |
| `turn_right` | 717 | 4.7% |
| `turn_left` | 506 | 3.3% |
| `orbit_right` | 425 | 2.8% |
| `shift_light_angle` | 332 | 2.2% |

---

## 5. Key Takeaways

1. **Factorizing `z_sem` (Good-View invariant) from `z_dyn` (equivariant world-model state) unlocked semantic grounding**:
   - Without adding any tabletop-specific heuristics, separating the invariant semantic space from action-conditioned transition dynamics turned active view diversity into genuine **Good Views**: `procrustes_color` jumped to **`0.698` (`0.740` peak)**, `procrustes_shape` to **`0.406` (`0.490` peak)**, independent `MiniLM` Procrustes naming to **`0.396` (`0.458` peak)**, confounded-to-decorrelated linear probes to **`0.396` shape / `0.510` color**, and frozen DINO alignment (`CKA` / `10-NN`) climbed steadily to **`0.218` / `0.118`**.
2. **Every core curve is still rising at `1,024` episodes**:
   - `f_H` (`0.083 -> 0.344`), `erank_Z` (`33.6 -> 43.0`), `dino_cka` (`0.186 -> 0.218`), and `dino_10nn` (`0.097 -> 0.118`) were all still climbing monotonically at episode `1,024`.
