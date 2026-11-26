"""投影头：TaxoBell §4.2/§4.3 的 box→Gaussian，纯 numpy 参考 + 确定性权重导出"""
import hashlib

import numpy as np

EPS = 1e-6
WEIGHT_KEYS = ("W1", "b1", "Wc", "bc", "Wo", "bo")


def init_weights(seed=42, k=8, hidden=64, d=8, scale=0.08):
    """确定性初始化。三后端必须调用本函数得到完全相同的 numpy 权重，禁止框架 RNG。"""
    g = np.random.default_rng(seed)

    def dense(inp, out):
        return (g.normal(0, scale, (inp, out)).astype(np.float32),
                np.zeros(out, dtype=np.float32))

    W1, b1 = dense(k, hidden)
    Wc, bc = dense(hidden, d)
    Wo, bo = dense(hidden, d)
    return dict(W1=W1, b1=b1, Wc=Wc, bc=bc, Wo=Wo, bo=bo)


def forward_numpy(w, n):
    """n [B,k] → (mu, o)，均 [B,d]，o > 0。relu 用 x*(x>0) 表达，与 ops 三后端一致。"""
    x = n.astype(np.float32)
    h = x @ w["W1"] + w["b1"]
    h = h * (h > 0)
    mu = h @ w["Wc"] + w["bc"]
    o = np.exp(h @ w["Wo"] + w["bo"]) + EPS
    return mu.astype(np.float32), o.astype(np.float32)


def weight_fingerprint(w):
    h = hashlib.sha256()
    for k in WEIGHT_KEYS:
        h.update(k.encode())
        h.update(np.ascontiguousarray(w[k], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]
