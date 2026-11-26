# 三论文合并主线：端到端全流程

主线：**TaxoBell 训练纵深 + PEEL 推理等价锚点 + SPARROW 评测协议桥**。
深挖一篇，另两篇只提供不重叠的补件，避免变成三篇泛讲。

## 1. 流程图

```
[L0 数据]  TaxoBell 5 数据集（Google Drive，预处理 .pkl）
           + PEEL wiki-ol / arxiv-ol + SPARROW CCS / Google / MeSH
           → 降级态先用 src/synth.py 的确定性合成树
           ↓ 文件 hash / 样本 id 集合逐一比对（L0 全字段逐字一致）
[L1 参考]  自建 PyTorch 对照：投影头（裸参数 2 层 MLP）→ Gaussian → 损失 → AdamW
           ↓ 同初始化 + 同负采样序列导出 .npz 交 Jittor（消除 RNG 不同源）
[L2 训练]  Jittor 主体：投影头 + BC/KL/hinge 全为逐元素原生算子（免 linalg.inv）
           ↓ scripts/train.py --backend both：逐 step loss 双曲线，相对误差 < 1e-3（前 20 步）
           ↓ --npz probe_input.npz：定向样本让 L_diverge/L_reg/L_clip 全程非零
           ↓ --loss official：第二套损失口径双跑；--steps 100：长程漂移标 DRIFT 不作门禁
[L3 前向]  逐项激活 max|Δ| < 1e-6（float32）；维度扫描逐项对表
           ↓
[锚点1 外] PEEL 作者 checkpoint `wiki_ol_probe.pt`（0.4MB）→ 推理逐元素对齐
[锚点2 内] 自建 PyTorch vs Jittor 同起点重训 → 证明训练路径等价（防两个一起错）
           ↓
[评测桥]   scripts/eval_pipeline.py：投影头前向 → rank_anchors 预测父边
           → SPARROW Node F1 / Edge F1 / Ancestor F1（纯 Python、无 LLM）
           + TaxoBell MR / MRR / R@k / H@k / Wu&P
           + PEEL OLLM 五指标（Graph F1 / Cont F1 / Fuzzy / Literal / Motif）
           ↓ 三套协议交叉验证；`--weights logs/trained_jittor.npz` 已 WSL 实跑（18:32），
             默认（无 --weights）为初始权重基线，两者分开落盘对比
           ↓ 边界：`perfect`/`broken` 是评测桥自检基线非模型产出；`mode_separability` 证明
             BC/KL 实现不同，但 `model_mode_disagreements` 仅 0/4→1/4，合成树上双协议几乎不可区分
[降级记账] 子集规模 / 冻结或缓存 BERT / epochs 缩减 / 未复现清单 / CPU 小时
[忠实性]   只比序关系不比绝对值：维度峰位、消融方向、预算不敏感符号
```

## 2. 逐段归属与分工

| 段 | 归属 | 实现位置 | 后端 |
|---|---|---|---|
| L0 数据 + 指纹 | TaxoBell §5 口径（20% 叶节点留出、保证金父留 seed） | `src/synth.py` | numpy / 标准库 |
| 编码器 | TaxoBell §4.2（`[CLS] e [SEP] s [SEP]` → `[CLS]`） | 降级为缓存 `.npy` | PyTorch / numpy 资产 |
| Box → Gaussian 投影 | TaxoBell §4.2/§4.3（2 个 2 层 MLP，`o=exp(·)`，`Σ=diag(o²)`） | `src/ops_*.py` | Jittor + PyTorch |
| 双能量 + 四损失 | TaxoBell §3/§4.4（式 2–13） | `src/ops_*.py` | Jittor + PyTorch |
| 排父推理 | TaxoBell §4.4（`TaxoBell_BC` / `TaxoBell_KL`） | `ops_numpy.rank_anchors` | Jittor + PyTorch |
| 外部锚点 | PEEL 作者 checkpoint | 待接入 | 纯 float / PyTorch |
| 评测桥 | SPARROW `eval.py` + TaxoBell 指标 | `scripts/eval_bridge.py` | 纯 Python / numpy |
| 模型产出→评测桥 | `rank_anchors` 预测父边接 Node/Edge/Ancestor F1 + MR/MRR | `scripts/eval_pipeline.py` | 纯 numpy（本地可跑） |
| 定向三项覆盖 | 放大 `n_*` 使 L_diverge/L_reg/L_clip 训练全程非零 | `scripts/make_probe.py` | 纯 numpy（本地可跑） |
| 投影头 + 训练 | TaxoBell §4.2/§4.3 + §4.4 | `src/projection.py` / `src/model_jittor.py` / `scripts/train.py` | Jittor 主体 + PyTorch 对照 |

**分工纪律**：训练与前向归 Jittor（含 `src/encoder_jittor.py` 复用的冻结 Jittor BERT 前向）；
数据、指标、评测、权重资产归 numpy / PyTorch / 纯 Python。
评测/图/IO 是「合规仪器层」，不是「用 NumPy 实现模型」（沿用旧项目 `10_jittor_boundary.md`）。
编码器前向可归 Jittor 的依据来自 WSL 旧仓库实测的纯 Jittor BERT-base，见 `jittor_vs_pytorch.md` 第 5 节。

## 3. TaxoBell 方法链（源码与论文双核对）

§3 预备：seed taxonomy `T0=(N0,E0)`，把查询概念挂到最合适的父（anchor）下；
box `b=(c,o)` 轴对齐超矩形；高斯为稳定取对角 `Σ=diag(σ²)`；两个能量
`BC=exp(-D_B)`（对称，式 2/3）、`D_KL`（非对称，式 4）。

| 步 | 实现 | 位置 |
|---|---|---|
| 1 编码 | `x=[CLS] e [SEP] s [SEP]`，BERT 最后层 `[CLS]` → `n∈R^k` | §4.2 |
| 2 Box 投影 | `c=MLP_c(n)`、`o=φ(MLP_o(n))`，φ=exp/softplus 保正；p/c/p' 各一个 box | 式 5/6 |
| 3 Gaussian 投影 | `μ=c`、`Σ=diag(o⊙o)`；`½log det Σ=Σlog oᵢ` 作可微体积代理 | §4.3 |
| 4 自监督构造 | 每条 `⟨child,anchor⟩` 为正；从 siblings/uncles/cousins/grandparents 挖 N 个 hard negatives | §4.4 |
| 5 损失 | 论文式 7–13；**官方源码是另一套**，见 `paper_vs_code.md` | 式 7–13 |
| 6 推理 | query → box → Gaussian → 对每个 anchor 打分，BC / -KL 两个排序变体；`o=k√Σ`, k∈{1,2,3} | §4.4 |

§5 评测：5 数据集（Env 261/261/深6、Sci 429/452/8、WordNet 20.5/19.5/3、
Food 1486/1533/8 多父、MeSH 9710/10498/12 多父）；20% 叶节点留出且保证金父留在 seed；
7 baselines；指标 MR↓ / MRR / R@k / H@k / Wu&P。消融：去掉 `L_sym` 掉得最狠
（SCI MRR +933% 差距），去掉 `L_asym` 也明显掉；Table 7 证明「先 box 再 Gaussian」优于直接投影。

**附录 A 训练配置**：48GB A6000；`bert-base-uncased`；投影头 2 个 2 层 MLP hidden 64 / dropout 0.2；
AdamW 端到端微调（BERT lr 9e-5 + 投影层 lr 1e-3，源码 `_select_optimizer` 把全部
`named_parameters()` 送入优化器）；batch 128、最多 125 epochs（README 默认 50，不一致）；
negatives Sci/Env 50、WordNet 10、MeSH/Food 20；权重 0.45/0.45/0.10，`λ=.3, C=1.5`。

## 4. 评测桥口径（P3）

- Ancestor F1 用 **macro**（逐节点祖先集合 F1 取平均）。实测 `micro` 对结构性碎片化
  **不敏感**（完美 1.0 / 碎片化仍 1.0），`macro` 敏感（1.0 → 0.9810），故选 macro。
- 根节点祖先集两边都空时算 **F1=1**（完全一致），否则会凭空把 macro 拉到 0.9714。
- 多父数据集（Food/MeSH）必须先定 **DAG→树转换口径**，否则 Ancestor F1 不可比。
- 本仓库 Wu&P 只实现「单条预测父边相对 gold 树的相似度，宏平均」，属**接口级**；
  论文 Wu&P 是整体 DAG 相似度。报告里必须标这个边界。
- 三篇论文的流程图**都不含评测模块**（PEEL 止于 Ontology DAG、SPARROW 止于 Unified
  taxonomy、TaxoBell 止于两个能量图）。含评测 + 双锚点的端到端总览图必须自己画。

## 5. 降级记账表模板（答辩用，逐项填真实值）

| 项 | 论文 | 本仓库降级值 | 影响 |
|---|---|---|---|
| 数据集 | 5 个官方 taxonomy | 合成树 35 节点 / 34 边 / 4 query | 只验等价性，不报绝对分 |
| 编码器 | 端到端微调 BERT | 冻结 + 缓存 `.npy`（合成态为哈希伪嵌入） | 训练纵深转为投影头 |
| epochs | 50（README）/ 125（附录） | **20 steps**（固定 batch，不是 20 epochs） | 只比 loss 曲线带宽与方向 |
| batch / dim / negs | 128 / 8 / 10–50 | 32 / 8 / 5 | 同上 |
| 硬件 | 48GB A6000 | CPU | 记录 CPU 小时 |
| 未复现清单 | — | 绝对分数、真实 BERT 前向、5 数据集全量、7 baselines | 明写，不藏 |

**耗时结构与「降级能否做到」**（`scripts/estimate_cost.py`，2026-11-25 实跑；源码实证见
`docs/paper_vs_code.md` §8）：最吃时间的是 **§4.2/附录 A 的 BERT 端到端微调**。
`_select_optimizer` 把全部 `named_parameters()` 送进 AdamW；`model.py` 的
`par/child_projection_box(method='normal')` 各调 `get_cls` **2 次**，`forward` 对
parent/child/neg 各一次 → **每 step 6 次 BERT 前向+反传**（`gated` 12 次）；
`padmaxlen=30`、`batch 128`、`epochs 50`、`drop_last=True`。

**修正此前口述**：「§4.4 每 epoch 重挖 hard negatives 图遍历」在官方源码里**不成立**。
`.pkl` 的 `train_child_parent_negative_parent_triple` 已离线展开，
`Data_TRAIN.__init__` 一次性读出 + `generate_all_token_ids` 一次性 tokenize，训练循环
只有 shuffle，没有重采样。实测 10k 节点重挖 50 次也仅 3.2s（`logs/negative_sampling.json`），
相对 BERT 微调可忽略；真正的第二耗时是每 epoch 的 `self.predict()`（test_loader
`batch=1` 全量候选打分）与 checkpoint I/O，二者未计入 FLOPs 主表，故预估是**下界**。

**19:16 WSL 补测标定**（`logs/bert_calibration.json`，本人执行）：batch 2/4/8/16/32 =
6.99/13.48/25.04/30.83/46.68G，单步 5.67/5.88/6.33/10.28/13.57s。19:02 只测到
batch8 的外推（`a=4.75+0.408·batch`）作废；补测后拟合收紧为
`t_step≈4.87s(固定) + 0.280s·batch`，固定开销占比随 batch 由 86%(b2) 降到 36%(b32)。

| 降级档 | 相对全量 | 实测外推 h | FLOPs 上界 h | 2h | 24h |
|---|---|---|---|---|---|
| 5 数据集 × 50ep | 1.0 | 12504.88 | 31060.15 | 否 | 否 |
| 5 数据集 × 125ep | 2.5 | 31262.19 | 77650.37 | 否 | 否 |
| Env × 50ep（b128） | 2.8e-2 | 346.41 | 860.44 | 否 | 否 |
| Env × 50ep（b32） | 2.8e-2 | 470.53 | 860.44 | 否 | 否 |
| Env × 50ep（b8） | 2.8e-2 | 967.01 | 860.44 | 否 | 否 |
| Env × 5ep × negs10 × b32 | 5.6e-4 | **9.46** | 17.29 | 否 | **是** |
| Env × 2ep × negs5 × b8 | 1.1e-4 | 3.89 | 3.46 | 否 | **是** |
| Env × 1ep × negs5 × b8（probe） | 5.6e-5 | **1.94** | 1.73 | **是** | **是** |
| probe gated（12 调用） | 1.1e-4 | 3.89 | 3.46 | 否 | **是** |
| 冻结/缓存 `[CLS]` × 20ep | 2.0e-8 | ≈0 | ≈0 | **是** | **是** |

两列口径：**实测外推**为主判据（标定 step-time × 该档 batch 与调用数），
**FLOPs 上界**为保守对照（统一 eff=25.04G）。`hours_measured` / `hours` 都在
`logs/cost_estimate.json`，`step_time_model.max_measured_batch=32`，更大 batch 才是外推。

**结论（三问合一，19:16 口径）**：
1. **2h 硬预算：结论翻转**。probe 档实测外推 **1.94h < 2h**（19:02 的 2.19h 作废），
   冻结档也达标 → 2 档可跑。这是补测大 batch 后 `b` 从 0.408 降到 0.280 的结果，
   不是放宽判据。仍要留余量：FLOPs 上界 1.73h，且每 epoch `predict()`/checkpoint I/O
   未计入，实跑可能略超。
2. **§4.4 图遍历**：可做，已做完。10k 节点全 epoch 重挖 3.39s = `9.43e-4h`，
   相对 BERT 差 7 个数量级；官方更是离线一次性、连这 3.39s 都不花，不是瓶颈。
3. **1 天（24h）容忍**：**能做** §4.4 采样 + probe（1.94h）+ gated（3.89h）+
   2ep（3.89h）+ **5ep/negs10（9.46h）** + 冻结档，共 5 档；
   **仍不能做** 全量 5 数据集（≥12504h）或任何 50ep 档（≥346h）。

**修正此前口述**：「降 batch 只降峰值内存、不省时间」应升级为「**降 batch 反而更亏**」——
FLOPs 总量近恒定（离散 2.68%），但 CPU 固定开销摊不薄：`Env×50ep` 从 b128 346.41h
恶化到 b8 967.01h（慢 2.79×）。省时间靠**升 batch（内存允许下）/ 降 epochs /
降数据集边数 / 降 negsamples / 降 seq_len**，最大杠杆仍是**冻结或缓存 `[CLS]`**
（6 次 BERT 调用 → 0，差约 8 个数量级）。

**梯度链路实证**（`logs/bert_probe.json`，19:41）：`scripts/train_bert_probe.py` 用随机
token ids 跑 Jittor `BertModel` 全参微调，`opt.backward(loss); opt.step()`，
`grad_chain_ok=true`、`param_max_delta=1.67e-6`、参数与梯度全有限 →
**BERT 端到端微调的梯度链路通**，不依赖权重/数据。数值口径注意：该文件是
`--batch 32`（第二次命令覆盖了默认 b8 输出），median `67.25s/step`（6 次 get_cls
合计）、`eff=56.53G`、164 步外推 3.06h；其 `target_steps=164` 取自 probe 档（b8），
与 b32 不是同一配置，故 **3.06h 不能当任一档正式结论**，只作链路实证与 b32 单步实测量。
脚本已加 `--tag` 分开落盘，避免再次覆盖。

真跑三要素已齐：`bert_jittor.npz`（权重转换完成）+ `vocab.txt` + 官方 Drive
原始数据生成的 5 份 pkl（environment/science/wordnet/semeval_food/mesh）+ 本地替代 pkl。
本仓库既有 PASS 只覆盖「投影头训练等价」，**不含 BERT 微调等价**，报告里不能混写。

**B 档前置下载（2026-11-25 21:57 完成）**：HF 直连 502，改走 `hf-mirror.com`，
四个文件落在 `data/external/bert-base-uncased/`（`data/` 已被 `.gitignore` 排除，不入库）：

| 文件 | 字节 | 校验 |
|---|---|---|
| `config.json` | 570 | `vocab_size=30522 / hidden=768 / 12 layers / 12 heads / gelu / eps=1e-12`，与旧仓 Jittor BERT 架构逐项吻合 |
| `vocab.txt` | 231508 | 30522 行，首 `[PAD]` |
| `tokenizer_config.json` | 48 | — |
| `model.safetensors` | 440449768 | header 24048B / 206 tensors / F32；首传 curl code 56 中断于 388MB，`curl -C -` 续传补齐，与 Content-Length 精确一致 |

SHA256：`config 7160e155…`、`vocab 07eced37…`、`tokenizer a025160e…`、
`model.safetensors 68d45e23…`。

**格式边界（已澄清并可转换，2026-11-25 22:46 实跑）**：`model.safetensors` 的
`__metadata__={"format":"pt"}`，206 个 F32 tensor，是 HF `BertForMaskedLM` 的完整
state_dict。与旧仓 `hf_loader.build_mapping()` 的 199 键差异只有三处：
① 带 `bert.` 前缀；② LayerNorm 用 `gamma/beta` 而非 `weight/bias`；
③ 混入 `cls.*`（MLM/NSP head 7 键）。**`bert.pooler.dense.{weight,bias}` 是存在的**，
`query/key/value/output` 键名与 Jittor 完全一致，shape 也逐项一致
（旧仓 `nn.Linear` 是 `[out,in]`，与 HF 相同，**不需要转置**）。
此前记的「无 pooler / 需补映射 / 需转置」是探测输出被截断导致的误判，作废。

`scripts/convert_bert_safetensors.py`（纯标准库，Windows 侧无 numpy 也能跑）只做
去前缀 → `gamma/beta` 改名 → 丢 `cls.*`，手工构造 `npy v1.0 + zip` 输出
`bert_jittor.npz`。本地已实跑：`tensors 206 → mapped 199`、
`missing=0 / extra=0 / shape_mismatch=0 / transposed=0`、
`bert_jittor.npz 437977820B`，`logs/bert_convert.json` 留档。
注意装载时 **`BertModel(vocab_size=30522)`**，不能用旧仓默认的 31116。

## 5b. B 档可落地脚本（2026-11-25 写完，等 WSL 实跑）

所有脚本不改旧仓任何文件；用 `importlib` 按路径直载
`specter-jittor/src/model/{bert,hf_loader}.py`，避开 `sys.path` 里两个 `src` 包冲突。
`encoder_jittor.py` 的 `sys.path.insert` + `from src.model.bert` 写法在该场景会命中
已缓存的本仓 `src`，故 B 档链路**不复用**它，改由新脚本直载。

| 文件 | 依赖 | 作用 | 本地状态 |
|---|---|---|---|
| `scripts/convert_bert_safetensors.py` | 纯标准库 | safetensors → `bert_jittor.npz`，199 键与 `build_mapping()` 逐项校验 | **已实跑** |
| `src/bert_tokenizer.py` | 纯标准库 | WordPiece，`[CLS] name [SEP] def [SEP]`，不依赖 transformers | **已自检** |
| `scripts/check_bert_load.py` | numpy + jittor | `load_npz()` 199 键 1:1、embed 逐位一致、冻结前向有限且行间可分 | 待 WSL 跑 |
| `scripts/train_bert_e2e.py` | numpy + jittor | BERT `[CLS]` → 双 MLP → 双损失 → 训练 → 评测桥 | 待 WSL 跑 |
| `scripts/make_local_pkl.py` | 纯标准库 | 官方 12 键 ID 三元组格式本地替代 pkl（真实概念名+人工定义） | **已实跑** |
| `run_b.sh` | WSL + conda jittor | 一键：前置核对 → 转换兜底 → 装载自检 → 两套损失 dry-run → 本地 pkl → 可选官方 pkl | 待 WSL 跑 |

**跑法**（WSL，本人执行，不装包）：
```bash
cd ~/taxobell_jittor
bash run_b.sh                                  # dry-run + 本地替代 pkl 全链路
bash run_b.sh /path/to/taxonomy_data_XX_.pkl  # 若已拿到官方 .pkl
```
`run_b.sh` 内 `cc_path=/usr/bin/g++-12`、`conda activate jittor`、`jt.flags.use_cuda=0`
均已设好。分步也可：
```bash
python scripts/make_local_pkl.py               # 生成 data/local/taxonomy_data_local_.pkl
python scripts/check_bert_load.py
python scripts/train_bert_e2e.py --steps 20 --loss paper    --tag dryrun20
python scripts/train_bert_e2e.py --steps 20 --loss official --tag dryrun20_official
python scripts/train_bert_e2e.py --pkl data/local/taxonomy_data_local_.pkl --steps 20 --tag localpkl20
python scripts/train_bert_e2e.py --finetune-bert --steps 2 --batch 2 --tag ft2
```

**口径边界（写进报告，不能混）**：
* 默认冻结 BERT：一次编码缓存 `[CLS]`，只训投影头；投影头输入维度从降级的 8
  改为真实 768（`projection.init_weights(k=768)`），输出 `d=8`。
* `--finetune-bert` 才把 199 个 BERT 参数 + 6 个投影头参数一起交给
  `jt.optim.AdamW`；CPU 约 5.7s/step（batch2），只建议 2 步链路实证。
* 训练态纪律：冻结前向是推理类才调 `bert.eval()` 关 dropout；
  **投影头与 BERT 微调全程 `train()`，绝不 `eval()`**。
* `dry-run` 用 `synth.build()` 合成树，文本真实过 BERT，但**不是论文训练**；
  不含官方 5 数据集 / 50 epochs，不比论文绝对分数。
* 训练后权重是 768 维输入，**不能**喂给 8 维降级口径的 `eval_pipeline.py`；
  评测指标内置在 `train_bert_e2e.py` 的 `evaluation` 字段。

**2026-11-26 官方 Google Drive 数据到手（本人下载 `data-20261126T012441Z-1-001.zip`，9.2MB→39MB）**：
解压到 `data/official_drive/data/`，41 文件含 environment / science / mesh /
semeval_food 四个数据集的**官方原始文件** + mesh 预测日志。zip **不含 `processed/*.pkl`**，
pkl 由 `scripts/run_official_preprocess.py` 驱动官方 `pre_process.py` 生成（未改官方逻辑，
seed=20 / `expID=0` / `negsamples=5`）：

| 数据集 | 路由 | 概念 | 三元组 | test | pkl |
|---|---|---|---|---|---|
| environment | `create_data` 单父 | 261 | 1013 | 52 | 1.3MB |
| science | `create_data` 单父 | 429 | 1771 | 85 | 0.8MB |
| wordnet | `create_data` 单父（源为 QuanTaxo 副本，Drive 未含） | 2310 | 30482 | 214 | 42MB |
| semeval_food | `create_multiparent_data` | 1486 | 6430 | 118 | 0.3MB |
| mesh | `create_multiparent_data` | 9443 | 40875 | 890 | 3.4MB |

五份 pkl 均过 `load_official_pkl()` + 评测桥冒烟。mesh 官方脚本自弃 ~100 个
不在 `concept_id` 的 test 词，属原版行为，不补。**loader 两处 bug 已修**：
① 单父 `path2root[v]=[v,parent,...,root]`，`[0]` 是自身、`[1]` 才是真父 —— 改为主用
`child_parent_pair`（multiparent 版 `path2root` 为空），消除自环边与 Ancestor F1 失真；
② 官方 `np.random.choice` 把 `np.int64` 直接塞进三元组，它不是 `int` 子类，会被
`str()` 注册成纯数字伪概念（mesh 曾虚增到 14810 个）—— `resolve()` 归一化 `np.integer`
后概念数与官方 `concept_set` 逐个吻合（wordnet +1 为官方 supernode）。

**仍未到手**：`wordnet_verb` / `computer_science` / `psychology` 的原始 `.terms`/`.taxo`
（若 Drive 另有分卷 zip，给我路径即可按 `create_multiparent_data` 同法补齐）。
`run_b.sh` 3c 步现在自动：生成 4 份 pkl → environment 与 semeval_food 各跑两套损失口径。
2. **本地替代 pkl 已生成**（2026-11-26 实跑 `scripts/make_local_pkl.py`）：
   `data/local/taxonomy_data_local_.pkl`，59 真实英文概念（WordNet gloss 风格
   人工定义）× 240 三元组 × 7 test query，12 键与官方一致；装载与评测桥
   （候选池 MR/MRR + 三 F1）已在本机冒烟 PASS。官方等价 pkl 到位后，本地替代
   仅作「最小受控样例」保留。
3. ~~`cls_parity()` 需 WSL 装 transformers 后另跑~~ **2026-11-26 已闭环**：WSL jittor
   环境本就有 torch 2.15.0+cpu / transformers 5.23.0（writing_rules.md 经文件系统核实，
   无需安装）；新增 `scripts/parity_bert.py`（run_b.sh 第 2b 步）—— Jittor 与
   PyTorch/HF 两框架装**同一份** `bert_jittor.npz`（199 键 strict 对齐 HF
   BertModel.state_dict），同一 tokenizer 输入比最后层 `[CLS]` 与 pooler，判据
   cos>=0.999 沿用旧仓 03_m1_align.py，产物 `logs/bert_parity.json`。加上 A 档
   L2/L3（投影头 Jittor↔PyTorch 训练曲线与权重等价，logs/parity.log），双框架
   对照覆盖「编码器前向 + 投影头训练」全链。
4. C 篇第三段「实验完整结果图表」按用户要求「评测最后定，先不写」，继续留空；
   现在 environment/science 真实数据在手，B 档跑完后即可填「链路实证 × 真实数据」
   两档，仍非论文 50ep 绝对分数。

旧仓 SciBERT/SPECTER 权重只能作降级占位，PPT/报告须标注「非 bert-base-uncased」；
本轮已换成真正的 `bert-base-uncased`，该占位不再是必须。

## 6. 必须预先知道的批判点（不要写进 PPT 当优点）

1. 官方仓库**无 LICENSE、无 checkpoint、数据在外链、模型路径写死 `/home/models/...`**，
   且需 A6000 微调 125 epochs —— CPU 全量不可行，降级是必选而非可选。
2. **PEEL checkpoint 是 wiki-ol 域**，与 TaxoBell 五数据集不同域：只能作方法论级外部锚点
   （证明「我们有能力做推理等价」），不能当 TaxoBell 的数值锚点。
3. 论文 §4.4 与官方 `model.py::forward` **不是同一套损失**（详见 `paper_vs_code.md`）。若按论文公式实现
   再声称「复现了官方代码」，会被翻。
4. 三个非确定性坑：float32 累积漂移、Hungarian / tie-break 平局次序、框架 RNG 不同源。
   前两个靠固定次序，第三个靠 `.npz` 同起点。
