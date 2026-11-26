"""Jittor BERT ↔ PyTorch/HF BERT 前向对照（§4.2，同权重源 bert_jittor.npz）"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("cc_path", "/usr/bin/g++-12")

EXT = os.path.join(ROOT, "data", "external", "bert-base-uncased")
DEFAULT_REPO = "/home/guo76/specter-jittor"

NAMES = ["machine learning", "taxonomy induction", "word embeddings",
         "attention mechanism", "hypernym prediction", "root node"]
DEFS = ["statistical models learned from data",
        "building a hierarchy of concepts from text",
        "distributed representations of words in a vocabulary",
        "a weighted mechanism over value vectors",
        "predicting the parent of a given concept",
        "the top of a taxonomy with no parent"]


def load_old(repo):
    def mod(name, rel):
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            raise SystemExit("缺旧仓 %s；用 --repo 指定 specter-jittor 路径" % p)
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    return (mod("tb_old_bert", "src/model/bert.py").BertModel,
            mod("tb_old_hf", "src/model/hf_loader.py").load_npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ext", default=EXT)
    ap.add_argument("--npz", default=os.path.join(EXT, "bert_jittor.npz"))
    ap.add_argument("--max-len", type=int, default=64)
    a = ap.parse_args()
    for p in (a.npz, os.path.join(a.ext, "vocab.txt"),
              os.path.join(a.ext, "config.json")):
        if not os.path.exists(p):
            print("缺 %s" % p)
            return 1

    import numpy as np
    import torch
    import jittor as jt
    from transformers import AutoTokenizer, AutoModel
    from src.encoder_jittor import cls_parity

    jt.flags.use_cuda = 0
    torch.set_num_threads(4)

    tok = AutoTokenizer.from_pretrained(a.ext, local_files_only=True)
    enc = tok(NAMES, DEFS, padding=True, truncation="only_second",
              max_length=a.max_len, return_tensors="np")
    ids = enc["input_ids"].astype("int32")
    mask = enc["attention_mask"].astype("float32")
    tt = enc["token_type_ids"].astype("int32")

    # PyTorch/HF 侧：config-only 初始化（目录无权重文件也能走），再从同 npz strict 装载
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.ext, local_files_only=True)
    ref = AutoModel.from_config(cfg).eval()
    with np.load(a.npz) as z:
        sd = {k: torch.from_numpy(np.asarray(z[k], np.float32)) for k in z.files}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[FAIL] npz->HF state_dict mismatch missing=%r unexpected=%r"
              % (missing[:5], unexpected[:5]))
        return 1

    # Jittor 侧：旧仓 BertModel + load_npz
    BertModel, load_npz = load_old(a.repo)
    model = BertModel(vocab_size=len(tok))
    load_npz(model, a.npz, verbose=False)
    model.eval()
    with jt.no_grad():
        h, pooled = model(jt.array(ids), jt.array(mask), jt.array(tt))
        jt.sync_all()
        cls_jt = np.ascontiguousarray(h[:, 0].numpy(), np.float32)
        pool_jt = np.ascontiguousarray(pooled.numpy(), np.float32)

    # HF 侧前向
    with torch.no_grad():
        out = ref(input_ids=torch.from_numpy(ids),
                  attention_mask=torch.from_numpy(mask),
                  token_type_ids=torch.from_numpy(tt),
                  output_hidden_states=True)
        cls_hf = out.hidden_states[-1][:, 0].numpy().astype(np.float32)
        pool_hf = out.pooler_output.numpy().astype(np.float32)

    r = cls_parity(cls_jt, cls_hf)
    rp = cls_parity(pool_jt, pool_hf)
    hf_layers = len(out.hidden_states) - 1          # 12 层（不含 embedding）
    ok = (r["cos_mean"] >= 0.999 and r["cos_min"] >= 0.999
          and r["max_abs_delta"] < 1e-3 and rp["cos_mean"] >= 0.999)
    print("== BERT 前向对照：Jittor ↔ PyTorch/HF（同 npz 权重，n=%d）==" % len(NAMES))
    print("  [CLS]  cos_mean=%.6f cos_min=%.6f max|Δ|=%.3e" %
          (r["cos_mean"], r["cos_min"], r["max_abs_delta"]))
    print("  pooler cos_mean=%.6f max|Δ|=%.3e  hf_layers=%d" %
          (rp["cos_mean"], rp["max_abs_delta"], hf_layers))
    print("  [PASS]" if ok else "  [FAIL]", "Jittor BERT ≡ PyTorch BERT 前向等价")
    rec = dict(n=len(NAMES), max_len=a.max_len, hf_layers=hf_layers,
               cls=r, pooler=rp, ok=ok,
               note=("两框架装同一 bert_jittor.npz（199 键 strict 对齐 HF "
                     "BertModel.state_dict），同一 tokenizer 输入；判据 "
                     "cos>=0.999 沿用旧仓 03_m1_align.py 口径"))
    with open(os.path.join(ROOT, "logs", "bert_parity.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("  -> logs/bert_parity.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
