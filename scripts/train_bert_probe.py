"""BERT 全参微调梯度链路 probe：实测 step time 外推 probe 档，不依赖权重/数据"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("SPECTER_JITTOR_REPO", "/home/guo76/specter-jittor")
OUTDIR = os.path.join(ROOT, "logs")
PARAMS = 110e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=31116)
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--target-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--day", type=float, default=24.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.join(OUTDIR, f"bert_probe{'_'+a.tag if a.tag else ''}.json")
    if a.target_steps <= 0:
        p = os.path.join(ROOT, "logs", "cost_estimate.json")
        a.target_steps = 164
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                ce = json.load(f)
            a.target_steps = next((t["total_steps"] for t in ce["tiers"]
                                   if t["name"] == "probe_1ep_negs5_batch8"), 164)
    os.environ.setdefault("cc_path", "/usr/bin/g++-12")
    import numpy as np
    import jittor as jt
    jt.flags.use_cuda = 0
    sys.path.insert(0, REPO)
    from src.model.bert import BertModel
    m = BertModel(vocab_size=a.vocab)
    m.train()
    params = m.parameters()
    opt = jt.optim.AdamW(params, lr=1e-5)
    rng = np.random.default_rng(42)
    ids = [jt.array(rng.integers(0, a.vocab, (a.batch, a.seq)).astype("int32"))
           for _ in range(a.calls)]
    mask = jt.array(np.ones((a.batch, a.seq), "int32"))
    tt = jt.array(np.zeros((a.batch, a.seq), "int32"))
    before = np.asarray(params[0].numpy()).copy()

    def step():
        hs = [m(i, mask, tt)[0][:, 0, :] for i in ids]
        loss = sum(h.mean() for h in hs) / a.calls
        opt.backward(loss)
        opt.step()
        jt.sync_all()

    for _ in range(a.warmup):
        step()
    ts = []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    after = np.asarray(params[0].numpy())
    gv = params[0].opt_grad(opt)
    g = np.asarray(gv.numpy()) if gv is not None else None
    proj = med * a.target_steps / 3600
    delta = float(np.abs(after - before).max())
    doc = dict(
        source="Jittor BertModel 随机初始化；6 次 get_cls 等价梯度链路（与权重/数据无关）",
        vocab=a.vocab, seq=a.seq, batch=a.batch, calls=a.calls, steps=a.steps,
        warmup=a.warmup, seconds=ts, median_step_s=med,
        eff_gflops=6.0 * PARAMS * a.batch * a.seq * a.calls / med / 1e9,
        target_steps=a.target_steps, projected_hours=proj,
        within_2h=bool(proj <= a.budget), within_day=bool(proj <= a.day),
        param_max_delta=delta, param_finite=bool(np.isfinite(after).all()),
        grad_finite=bool(np.isfinite(g).all()) if g is not None else None,
        grad_chain_ok=bool(g is not None and delta > 0 and np.isfinite(after).all()
                           and np.isfinite(g).all()),
        note="opt.backward(loss); opt.step(); jt.sync_all()；训练态 m.train()，绝不 eval()")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"== BERT 梯度链路 probe == median={med:.2f}s/step  "
          f"外推 {a.target_steps} 步={proj:.2f}h  2h={'Y' if doc['within_2h'] else '-'}"
          f"/{a.day}h={'Y' if doc['within_day'] else '-'}")
    print(f"  梯度链路 ok={doc['grad_chain_ok']} param_delta={delta:.3e} "
          f"param_finite={doc['param_finite']} grad_finite={doc['grad_finite']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
