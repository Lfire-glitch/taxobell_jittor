# taxobell_jittor

## 项目说明

TaxoBell 论文的 Jittor 复现：Gaussian box（μ 位置 / Σ=diag(o²)）+ BC/KL 双能量 + 硬负例，
配 SPARROW 的 Node/Edge/Ancestor F1 评测协议。核心目标不是追论文绝对分数，而是
**Jittor ↔ PyTorch 双框架逐项等价**（同 seed 同输入下 loss 曲线、权重、BERT 前向全部对齐），
并在官方数据子集上做端到端链路实证。

- 训练与前向归 Jittor；PyTorch 只作数值对照（同一 npz 权重、同一输入）。
- 数值口径三后端统一：`CLAMP_EPS=1e-4`，`_safe_log_arg(x)=relu(x-1e-4)+1e-4`。
- 官方 5 数据集由原始 `.taxo/.terms` 经官方 `pre_process.create_data()` 原样生成 pkl。
- 一键入口 `run_b.sh`：前置核对 → BERT 装载 → 双框架对照 → dry-run → 官方 pkl 评测。
- 所有实验证据在 `logs/*.json`，可逐条复验。

## 项目目录

```
taxobell_jittor/
  README.md  run_b.sh
  src/         源码：算子三后端 / 投影头 / BERT 编码 / 数据与负例
  scripts/     实验脚本：自检 / parity / 训练 / 数据预处理 / 评测
  data/        external BERT 权重、官方 pkl、本地替代 pkl、官方原始数据
  logs/        全部实验证据 *.json
  docs/        mainline.md（主线全景）
  config/      default.json 实验配置
  chk/official/  官方 GitHub 源码留档（结构实证用）
```

## 每个文件作用

### src/

| 文件 | 作用 |
|---|---|
| `ops_numpy.py` | TaxoBell §3/§4 对角高斯闭式算子的纯 numpy 参考实现 |
| `ops_torch.py` | PyTorch 参考实现，与 ops_numpy.py 逐字对齐 |
| `ops_jittor.py` | Jittor 实现，与上两者逐字对齐（L2 训练等价目标） |
| `projection.py` | 双 MLP 投影头（box→Gaussian）numpy 参考 + 确定性权重导出 |
| `model_jittor.py` | Jittor 主体：投影头 → (mu, o) → 损失 → 训练 step |
| `model_pytorch.py` | PyTorch 对照实现，供 Jittor 主体逐项对照 |
| `encoder_jittor.py` | 冻结 BERT 的 Jittor 前向 ↔ HF/PyTorch 逐层对照 |
| `bert_tokenizer.py` | WordPiece tokenizer，纯标准库不依赖 transformers |
| `synth.py` | 确定性合成 mini-taxonomy 生成器（L0 数据层） |
| `hard_negatives.py` | §4.4 四类硬负例（siblings/uncles/cousins/grandparents） |

### scripts/

| 文件 | 作用 |
|---|---|
| `selfcheck_ops.py` | 算子级公式自检（纯 numpy） |
| `selfcheck_numeric.py` | float32 数值稳定性回归自检 |
| `selfcheck_pipeline.py` | 全链路自检：投影头→损失→排父→评测桥 |
| `selfcheck_encoder.py` | 编码器层契约自检 |
| `parity.py` | L3 前向双框架对齐，`--emit-input` 生成 L0 固定输入 |
| `parity_bert.py` | Jittor BERT ↔ PyTorch/HF BERT 前向对照（同 npz 权重源） |
| `train.py` | L2 训练对齐：Jittor 主体 ↔ PyTorch 对照 |
| `train_bert_probe.py` | BERT 全参微调梯度链路 probe 与耗时实测 |
| `train_bert_e2e.py` | B 档端到端：BERT [CLS] → 双 MLP → 双损失 → 训练 → 评测桥 |
| `estimate_cost.py` | BERT 端到端微调耗时预估与降级档位判定 |
| `calibrate_bert.py` | Jittor BERT 单步 fwd+bwd 稳态耗时标定 |
| `make_probe.py` | 生成定向输入使 L_diverge/L_reg/L_clip 全程非零 |
| `convert_bert_safetensors.py` | HF safetensors → bert_jittor.npz（199 键 strict） |
| `check_bert_load.py` | 验证 npz 1:1 装载 + BERT [CLS] 前向有限 |
| `run_official_preprocess.py` | 驱动官方 pre_process 生成 processed/*.pkl（不改官方逻辑） |
| `make_local_pkl.py` | 本地替代 pkl 生成器（官方 12 键结构） |
| `eval_bridge.py` | 评测协议桥：SPARROW 三 F1 + 排序指标 |
| `eval_pipeline.py` | 模型产出 → 评测桥：前向 → rank_anchors → 全指标 |

### data/ 与 logs/

| 路径 | 作用 |
|---|---|
| `data/external/bert-base-uncased/` | BERT 权重（vocab.txt / config.json / npz） |
| `data/{ds}/processed/taxonomy_data_05_.pkl` | 官方等价 pkl：environment / science / wordnet / semeval_food / mesh |
| `data/local/` | 本地替代 pkl（真实概念名 + 人工定义） |
| `data/official_raw/` | QuanTaxo 原始数据（官方预处理输入格式） |
| `logs/*.json` | 自检、parity、训练曲线、评测指标、BERT 对照的全部证据 |

## 完整流程

```bash
# 0. 环境：WSL conda env jittor（jittor + torch + transformers，CPU 口径）
cd ~/taxobell_jittor

# 1. A 档：三后端算子自检 → 训练等价 → 权重等价
python scripts/selfcheck_ops.py
python scripts/train.py            # L2：Jittor ↔ PyTorch 同 seed 20 步
python scripts/parity.py           # L3：训练后权重 max|Δ| < 1e-6

# 2. B 前置：safetensors → npz → 装载自检
python scripts/convert_bert_safetensors.py
python scripts/check_bert_load.py

# 3. BERT 双框架对照（同 npz、同输入，cos ≥ 0.999）
python scripts/parity_bert.py

# 4. 官方数据：原始 taxo → 官方 create_data() → pkl
python scripts/run_official_preprocess.py --datasets environment science semeval_food mesh

# 5. B 档端到端：冻结 BERT 编码 → 投影头训练 → 排序 + 三 F1 评测
python scripts/train_bert_e2e.py --pkl data/environment/processed/taxonomy_data_05_.pkl \
    --steps 20 --loss paper  --tag envpkl20
python scripts/train_bert_e2e.py --pkl data/environment/processed/taxonomy_data_05_.pkl \
    --steps 20 --loss official --tag envpkl20_official

# 或一键执行 2–5：
bash run_b.sh
```

产物对照：`logs/train_parity.json`（A 档曲线）、`logs/bert_parity.json`（BERT cos）、
`logs/b_e2e_*.json`（各数据集 × 双损失的 MR/MRR 与 Node/Edge/Ancestor F1）。
