# taxobell_jittor

## Project Overview

A Jittor reproduction of the TaxoBell paper: Gaussian boxes (μ location / Σ=diag(o²)) with
BC/KL dual energies and hard negatives, evaluated under the SPARROW protocol
(Node/Edge/Ancestor F1). The goal is not to chase the paper's absolute scores but
**item-by-item Jittor ↔ PyTorch equivalence** (identical loss curves, weights, and BERT
forward outputs under the same seed and inputs), plus an end-to-end pipeline validation
on official data subsets.

- Training and forward pass run in Jittor; PyTorch serves only as the numerical reference
  (same npz weights, same inputs).
- A unified numerical convention across all three backends: `CLAMP_EPS=1e-4`,
  `_safe_log_arg(x)=relu(x-1e-4)+1e-4`.
- The official 5 datasets are produced from raw `.taxo/.terms` files by the official
  `pre_process.create_data()` unchanged.
- One-entry runner `run_b.sh`: preflight checks → BERT loading → dual-framework parity
  → dry-run → official pkl evaluation.
- All experimental evidence lives in `logs/*.json`, independently re-verifiable.

## Repository Layout

```
taxobell_jittor/
  README.md  run_b.sh
  src/           source: three-backend ops / projection head / BERT encoder / data & negatives
  scripts/       experiments: self-checks / parity / training / data preprocessing / evaluation
  data/          external BERT weights, official pkls, local substitute pkl, official raw data
  logs/          all experimental evidence (*.json)
  docs/          mainline.md (project mainline)
  config/        default.json experiment config
  chk/official/  archived official GitHub sources (used for structure verification)
```

## File Reference

### src/

| File | Purpose |
|---|---|
| `ops_numpy.py` | Pure-numpy reference implementation of TaxoBell §3/§4 diagonal-Gaussian closed forms |
| `ops_torch.py` | PyTorch reference implementation, line-by-line aligned with ops_numpy.py |
| `ops_jittor.py` | Jittor implementation, line-by-line aligned with the above (L2 training-equivalence target) |
| `projection.py` | Double-MLP projection head (box→Gaussian), numpy reference + deterministic weight export |
| `model_jittor.py` | Jittor main model: projection head → (mu, o) → loss → training step |
| `model_pytorch.py` | PyTorch counterpart for item-by-item comparison with the Jittor model |
| `encoder_jittor.py` | Frozen-BERT Jittor forward ↔ HF/PyTorch layer-wise comparison |
| `bert_tokenizer.py` | WordPiece tokenizer, pure standard library, no transformers dependency |
| `synth.py` | Deterministic synthetic mini-taxonomy generator (L0 data layer) |
| `hard_negatives.py` | §4.4 hard negatives (siblings/uncles/cousins/grandparents) |

### scripts/

| File | Purpose |
|---|---|
| `selfcheck_ops.py` | Operator-level formula self-check (pure numpy) |
| `selfcheck_numeric.py` | float32 numerical-stability regression self-check |
| `selfcheck_pipeline.py` | Full-pipeline self-check: projection head → loss → parent ranking → eval bridge |
| `selfcheck_encoder.py` | Encoder-layer contract self-check |
| `parity.py` | L3 dual-framework forward alignment; `--emit-input` emits the L0 fixed input |
| `parity_bert.py` | Jittor BERT ↔ PyTorch/HF BERT forward comparison (same npz weight source) |
| `train.py` | L2 training alignment: Jittor main ↔ PyTorch reference |
| `train_bert_probe.py` | Full fine-tuning gradient-path probe with measured step time |
| `train_bert_e2e.py` | Tier-B end-to-end: BERT [CLS] → double MLP → dual loss → training → eval bridge |
| `estimate_cost.py` | Fine-tuning cost estimate and degradation-tier decision |
| `calibrate_bert.py` | Steady-state per-step Jittor BERT fwd+bwd timing calibration |
| `make_probe.py` | Generates directed inputs keeping L_diverge/L_reg/L_clip nonzero throughout |
| `convert_bert_safetensors.py` | HF safetensors → bert_jittor.npz (199 keys, strict) |
| `check_bert_load.py` | Verifies 1:1 npz loading + finite BERT [CLS] forward |
| `run_official_preprocess.py` | Drives the official pre_process to emit processed/*.pkl (no logic changes) |
| `make_local_pkl.py` | Local substitute pkl generator (official 12-key structure) |
| `eval_bridge.py` | Evaluation protocol bridge: SPARROW three F1s + ranking metrics |
| `eval_pipeline.py` | Model output → eval bridge: forward → rank_anchors → all metrics |

### data/ and logs/

| Path | Purpose |
|---|---|
| `data/external/bert-base-uncased/` | BERT weights (vocab.txt / config.json / npz) |
| `data/{ds}/processed/taxonomy_data_05_.pkl` | Official-equivalent pkls: environment / science / wordnet / semeval_food / mesh |
| `data/local/` | Local substitute pkl (real concept names + authored definitions) |
| `data/official_raw/` | QuanTaxo raw data (official preprocessing input format) |
| `logs/*.json` | All evidence: self-checks, parity, training curves, metrics, BERT comparison |

## Full Workflow

```bash
# 0. Environment: WSL conda env jittor (jittor + torch + transformers, CPU mode)
cd ~/taxobell_jittor

# 1. Tier A: three-backend op self-checks → training equivalence → weight equivalence
python scripts/selfcheck_ops.py
python scripts/train.py            # L2: Jittor ↔ PyTorch, same seed, 20 steps
python scripts/parity.py           # L3: post-training weights max|Δ| < 1e-6

# 2. Tier-B prerequisites: safetensors → npz → load check
python scripts/convert_bert_safetensors.py
python scripts/check_bert_load.py

# 3. Dual-framework BERT comparison (same npz, same inputs, cos ≥ 0.999)
python scripts/parity_bert.py

# 4. Official data: raw taxo → official create_data() → pkl
python scripts/run_official_preprocess.py --datasets environment science semeval_food mesh

# 5. Tier B end-to-end: frozen-BERT encoding → projection head training → ranking + three F1s
python scripts/train_bert_e2e.py --pkl data/environment/processed/taxonomy_data_05_.pkl \
    --steps 20 --loss paper  --tag envpkl20
python scripts/train_bert_e2e.py --pkl data/environment/processed/taxonomy_data_05_.pkl \
    --steps 20 --loss official --tag envpkl20_official

# Or run steps 2–5 with one command:
bash run_b.sh
```

Artifacts: `logs/train_parity.json` (Tier-A curves), `logs/bert_parity.json` (BERT cos),
`logs/b_e2e_*.json` (MR/MRR and Node/Edge/Ancestor F1 per dataset × loss variant).
