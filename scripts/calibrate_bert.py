"""WSL 标定：Jittor BERT 单步 fwd+bwd 稳态耗时 → eff_gflops，供 estimate_cost.py 回填"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("SPECTER_JITTOR_REPO", "/home/guo76/specter-jittor")
OUT = os.path.join(ROOT, "logs", "bert_calibration.json")
PARAMS = 110e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--vocab", type=int, default=31116)
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--batches", default="2,4,8")
    a = ap.parse_args()
    os.environ.setdefault("cc_path", "/usr/bin/g++-12")
    import numpy as np
    import jittor as jt
    jt.flags.use_cuda = 0
    sys.path.insert(0, REPO)
    from src.model.bert import BertModel
    m = BertModel(vocab_size=a.vocab)
    if a.weights:
        from src.model.hf_loader import load_npz
        load_npz(m, a.weights, verbose=False)
    m.train()
    opt = jt.optim.AdamW(m.parameters(), lr=1e-5)
    rows = {}
    rng = np.random.default_rng(42)
    for b in [int(x) for x in a.batches.split(",")]:
        ids = jt.array(rng.integers(0, a.vocab, (b, a.seq)).astype("int32"))
        mask = jt.array(np.ones((b, a.seq), "int32"))
        tt = jt.array(np.zeros((b, a.seq), "int32"))

        def step():
            h, _ = m(ids, mask, tt)
            opt.backward(h[:, 0, :].mean())
            opt.step()
            jt.sync_all()

        for _ in range(a.warmup):
            step()
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            step()
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        g = 6.0 * PARAMS * b * a.seq / med / 1e9
        rows[str(b)] = dict(seconds=ts, median_s=med, eff_gflops=g)
    g = statistics.median(v["eff_gflops"] for v in rows.values())
    doc = dict(source="Jittor BertModel 随机初始化（架构级标定，与权重无关）"
               if not a.weights else f"Jittor BertModel + {a.weights}",
               vocab=a.vocab, seq=a.seq, batches=rows, eff_gflops=g,
               note="warmup 排除首次 JIT 编译；训练态 m.train()，绝不 eval()")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"== Jittor BERT 标定 ==  eff_gflops={g:.2f}")
    for b, v in rows.items():
        print(f"  batch={b:>3s} median={v['median_s']:.3f}s eff={v['eff_gflops']:.2f}G")
    print(f"  -> {OUT}")
    print(f"  回填：python scripts/estimate_cost.py --calibrate-json {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
