"""§4.4 hard negatives：siblings/uncles/cousins/grandparents 采样与重挖计时"""
import json
import os
import random
import time
from collections import defaultdict

KINDS = ("siblings", "uncles", "cousins", "grandparents")


def index(edges):
    par = defaultdict(list)
    ch = defaultdict(list)
    for c, p in edges:
        par[c].append(p)
        ch[p].append(c)
    return par, ch


def sample(edges, kinds=KINDS):
    par, ch = index(edges)
    out = {}
    for c, p in edges:
        gps = par[p]
        sib = [x for x in ch[p] if x != c]
        unc = [u for g in gps for u in ch[g] if u != p]
        cou = [x for u in unc for x in ch[u]]
        cand = dict(siblings=sib, uncles=unc, cousins=cou, grandparents=gps)
        out[c] = {k: cand[k] for k in kinds}
    return out


def random_tree(n, seed=42, span=20):
    r = random.Random(seed)
    return [(c, r.randrange(max(0, c - span), c)) for c in range(1, n)]


def timed(edges, epochs=50):
    t0 = time.perf_counter()
    s = sample(edges)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(epochs):
        sample(edges)
    tn = time.perf_counter() - t0
    return dict(n_nodes=len({x for e in edges for x in e}), n_edges=len(edges),
                epochs=epochs, sample_once_s=t1, resample_all_s=tn,
                cached_all_s=t1, resample_over_cache_x=tn / t1 if t1 > 0 else None,
                counts={k: sum(len(v[k]) for v in s.values()) for k in KINDS})


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import synth
    cases = dict(synth=synth.build().edges, mesh_scale=random_tree(10000))
    res = {k: timed(v) for k, v in cases.items()}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "logs", "negative_sampling.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(source="§4.4 hard negatives；官方 .pkl 已离线展开，见 docs/paper_vs_code.md §8",
                       cases=res), f, indent=2, ensure_ascii=False)
    print("== §4.4 负样本采样 / 每 epoch 重挖 ==")
    for k, v in res.items():
        print(f"  {k:11s} nodes={v['n_nodes']:6d} edges={v['n_edges']:6d} "
              f"once={v['sample_once_s']:.4f}s x{v['epochs']}={v['resample_all_s']:.3f}s "
              f"cache={v['cached_all_s']:.4f}s  ratio={v['resample_over_cache_x']:.1f}x")
        print(f"    {v['counts']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
