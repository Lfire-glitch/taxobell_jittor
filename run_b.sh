#!/usr/bin/env bash
# ============================================================================
# B 档实验 runbook —— 明天在 WSL 里本人执行：bash run_b.sh [processed.pkl 路径]
# 不装包 / 不建环境 / 不改系统配置；所有产物落 logs/。
# 前置（Windows 侧已完成）：data/external/bert-base-uncased/{vocab.txt,bert_jittor.npz}
# 已经 9P 同步到 WSL 原生 ext4。safetensors 只在 Windows 侧保留，WSL 不需要。
# ============================================================================
set -euo pipefail

export cc_path=/usr/bin/g++-12   # 非交互 shell 不读 .bashrc；g++-15 会让 jittor import 失败
for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$c" ]; then . "$c"; break; fi
done
conda activate jittor

REPO="$HOME/taxobell_jittor"
cd "$REPO" || { echo "仓库不在 $REPO，先在 Windows 侧跑 sync_to_wsl.sh"; exit 1; }
EXT="data/external/bert-base-uncased"
PKL="${1:-}"

echo "== 0. 前置核对 =="
python - <<'PY'
import importlib
for m in ("numpy", "jittor", "torch", "transformers"):
    try:
        mod = importlib.import_module(m)
        print("  %-13s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-13s MISSING -> %s" % (m, type(e).__name__))
PY
for f in vocab.txt bert_jittor.npz config.json tokenizer_config.json; do
    [ -f "$EXT/$f" ] || { echo "缺 $EXT/$f：Windows 侧跑 scripts/convert_bert_safetensors.py 后同步"; exit 1; }
done
ls -l "$EXT"/{vocab.txt,bert_jittor.npz}

echo "== 1. 权重转换（npz 已存在则跳过）=="
if [ -f "$EXT/model.safetensors" ] && [ ! -f "$EXT/bert_jittor.npz" ]; then
    python scripts/convert_bert_safetensors.py
else
    echo "  用已同步的 bert_jittor.npz"
fi

echo "== 2. 装载 + 冻结前向自检 =="
python scripts/check_bert_load.py

echo "== 2b. Jittor ↔ PyTorch/HF BERT 前向对照（同 npz 权重，§4.2 等价证据）=="
python scripts/parity_bert.py

echo "== 3. dry-run 端到端（冻结 BERT，只训投影头；两套损失口径）=="
python scripts/train_bert_e2e.py --steps 20 --loss paper    --tag dryrun20
python scripts/train_bert_e2e.py --steps 20 --loss official --tag dryrun20_official

echo "== 3b. 本地替代 pkl（官方 12 键 ID 三元组格式；无官方数据时的实跑数据）=="
LPKL="data/local/taxonomy_data_local_.pkl"
[ -f "$LPKL" ] || python scripts/make_local_pkl.py
python scripts/train_bert_e2e.py --pkl "$LPKL" --steps 20 --loss paper    --tag localpkl20
python scripts/train_bert_e2e.py --pkl "$LPKL" --steps 20 --loss official --tag localpkl20_official

echo "== 3c. 官方 pkl（Google Drive 原始 data/ + 官方 pre_process，免翻墙）=="
if [ -f data/official_drive/data/environment/environment_raw_en.taxo ]; then
    python scripts/run_official_preprocess.py --datasets environment science semeval_food mesh
    for ds in environment semeval_food; do
        python scripts/train_bert_e2e.py --pkl "data/$ds/processed/taxonomy_data_05_.pkl" \
            --steps 20 --loss paper --tag "${ds}pkl20"
        python scripts/train_bert_e2e.py --pkl "data/$ds/processed/taxonomy_data_05_.pkl" \
            --steps 20 --loss official --tag "${ds}pkl20_official"
    done
    # mesh 9.4k 概念 / wordnet 3.7k 概念，CPU 编码较慢，按需要手动跑：
    # python scripts/train_bert_e2e.py --pkl data/mesh/processed/taxonomy_data_05_.pkl --steps 20 --tag meshpkl20
    # python scripts/run_official_preprocess.py --datasets wordnet
    # python scripts/train_bert_e2e.py --pkl data/wordnet/processed/taxonomy_data_05_.pkl --steps 20 --tag wnpkl20
else
    echo "  跳过：data/official_drive 未同步（Windows 侧解压官方 zip 后跑 sync_to_wsl.sh）"
fi

echo "== 4. 官方 pkl 真实数据（可选）=="
if [ -n "$PKL" ] && [ -f "$PKL" ]; then
    python scripts/train_bert_e2e.py --pkl "$PKL" --steps 20 --loss paper --tag pkl20
else
    echo "  跳过：未提供官方 processed/taxonomy_data_*.pkl（Google Drive 未到手）"
    echo "  用法：bash run_b.sh /path/to/taxonomy_data_XX_.pkl"
    echo "  解析分支已按 chk/official/datasets.py 实证结构实现；若仍报无法识别，把 keys 贴给智能体"
fi

echo "== 5. BERT 全参微调（可选；CPU 约 5.7s/step@batch2，默认不跑）=="
# python scripts/train_bert_e2e.py --finetune-bert --steps 2 --batch 2 --tag ft2

echo "== 6. 产物 =="
ls -lat logs/bert_convert.json logs/bert_load_check.json logs/bert_parity.json logs/b_e2e_*.json 2>/dev/null || true
echo "完成。A 档 L2/L3 投影头等价（logs/parity.log、jittor_vs_pytorch.md）+ 2b BERT 前向等价"
echo "= Jittor↔PyTorch 双框架对照全链；C 篇评测可直接引用 logs/bert_parity.json。"
