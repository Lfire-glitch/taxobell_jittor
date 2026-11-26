"""WSL：验证 bert_jittor.npz 能被旧仓 load_npz() 1:1 装载 + BERT [CLS] 前向有限"""
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
    ap.add_argument("--npz", default=os.path.join(EXT, "bert_jittor.npz"))
    ap.add_argument("--vocab", default=os.path.join(EXT, "vocab.txt"))
    ap.add_argument("--seq-len", type=int, default=32)
    a = ap.parse_args()
    for p in (a.npz, a.vocab, os.path.join(a.repo, "src/model/bert.py")):
        if not os.path.exists(p):
            print("缺 %s" % p)
            return 1

    import numpy as np
    from src.bert_tokenizer import BertTokenizer
    import jittor as jt
    jt.flags.use_cuda = 0
    BertModel, load_npz = load_old(a.repo)

    tok = BertTokenizer(a.vocab, max_len=a.seq_len)
    model = BertModel(vocab_size=tok.vocab_size)
    loaded, skipped, transposed = load_npz(model, a.npz, verbose=True)
    print("== BERT 权重装载 ==")
    print("  vocab_size=%d loaded=%d skipped=%d transposed=%d"
          % (tok.vocab_size, loaded, len(skipped), len(transposed)))
    if loaded != 199 or transposed:
        print("  [WARN] 期望 loaded=199 / transposed=0，检查 convert 脚本与 vocab_size")
    with np.load(a.npz) as z:
        ref = z["embeddings.word_embeddings.weight"]
    got = np.asarray(model.embeddings.word_embeddings.weight.numpy(), np.float32)
    embed_delta = float(np.abs(got - ref).max())

    names = ["machine learning", "taxonomy induction",
             "word embeddings", "attention"]
    defs = ["statistical models learned from data",
            "building a hierarchy of concepts",
            "distributed representations of words",
            "a weighted sum over values"]
    ids, mask, tt = tok.encode_batch(names, defs)
    model.eval()                      # 推理类才 eval()；此处纯冻结前向
    with jt.no_grad():
        h, pooled = model(jt.array(np.asarray(ids, np.int32)),
                          jt.array(np.asarray(mask, np.float32)),
                          jt.array(np.asarray(tt, np.int32)))
        jt.sync_all()
        cls = np.asarray(h[:, 0].numpy(), np.float32)
        pool = np.asarray(pooled.numpy(), np.float32)
    off = np.abs(cls[:, None] - cls[None, :]).sum(-1)
    off += np.eye(len(cls), dtype=np.float32) * 1e3
    distinct = bool(off.min() > 1e-4)
    finite = bool(np.isfinite(cls).all() and np.isfinite(pool).all())
    ok = (loaded == 199 and not transposed and embed_delta < 1e-6
          and finite and distinct and cls.shape == (4, 768)
          and bool(np.all(np.abs(pool) <= 1.0 + 1e-5)))
    print("  embed_max_delta=%.3e  cls=%s  finite=%s  distinct=%s  pooled|max|=%.4f"
          % (embed_delta, cls.shape, finite, distinct, float(np.abs(pool).max())))
    print("  [PASS]" if ok else "  [FAIL]", "bert-base-uncased → Jittor BertModel")
    rec = dict(npz=os.path.relpath(a.npz, ROOT).replace("\\", "/"),
               vocab_size=tok.vocab_size, loaded=loaded,
               skipped=len(skipped), transposed=len(transposed),
               embed_max_delta=embed_delta, cls_shape=list(cls.shape),
               cls_norm_mean=float(np.linalg.norm(cls, axis=1).mean()),
               pooled_abs_max=float(np.abs(pool).max()),
               finite=finite, distinct=distinct, ok=ok,
               note=("Jittor 侧装载与冻结前向自检；未做 HF/transformers 逐层对照"
                     "（WSL 无 transformers），cls_parity 需另装环境后补"))
    path = os.path.join(ROOT, "logs", "bert_load_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("  -> logs/bert_load_check.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
