"""算子级公式自检：只依赖 numpy，Windows / WSL 都能跑"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import ops_numpy as O  # noqa: E402

RNG = np.random.default_rng(42)
B, D = 8, 8
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


# ---------- 全矩阵参考（只用于自检，不进仓库算子层） ----------
def _sig(o):
    v = (o + O.EPS) ** 2
    return np.eye(o.shape[-1]) * v[:, None, :]


def kl_full(mu1, o1, mu2, o2):
    S1, S2 = _sig(o1), _sig(o2)
    S2i = np.linalg.inv(S2)
    d = mu1.shape[-1]
    tr = np.trace(S2i @ S1, axis1=-2, axis2=-1)
    diff = (mu2 - mu1)[..., None]
    mah = (diff.transpose(0, 2, 1) @ S2i @ diff).squeeze(-1).squeeze(-1)
    _, ld2 = np.linalg.slogdet(S2)
    _, ld1 = np.linalg.slogdet(S1)
    return 0.5 * (tr + mah - d + ld2 - ld1)


def bc_full(mu1, o1, mu2, o2):
    S1, S2 = _sig(o1), _sig(o2)
    Sm = 0.5 * (S1 + S2)
    Smi = np.linalg.inv(Sm)
    diff = (mu1 - mu2)[..., None]
    t1 = 0.125 * (diff.transpose(0, 2, 1) @ Smi @ diff).squeeze(-1).squeeze(-1)
    _, ldm = np.linalg.slogdet(Sm)
    _, ld1 = np.linalg.slogdet(S1)
    _, ld2 = np.linalg.slogdet(S2)
    return np.exp(-(t1 + 0.5 * (ldm - 0.5 * (ld1 + ld2))))


def rand_gauss(scale=1.0):
    mu = RNG.normal(0, 0.6, (B, D))
    o = np.exp(RNG.normal(0, 0.35, (B, D))) * scale
    return mu, o


def main():
    print("== TaxoBell 对角高斯闭式算子自检 ==")
    mu1, o1 = rand_gauss()
    mu2, o2 = rand_gauss()

    check("相同高斯 BC=1", np.allclose(O.bc_diag(mu1, o1, mu1, o1), 1.0, atol=1e-10),
          f"max|BC-1|={np.max(np.abs(O.bc_diag(mu1,o1,mu1,o1)-1)):.2e}")
    check("相同高斯 KL=0", np.allclose(O.kl_diag(mu1, o1, mu1, o1), 0.0, atol=1e-10),
          f"max|KL|={np.max(np.abs(O.kl_diag(mu1,o1,mu1,o1))):.2e}")
    check("KL >= 0", np.all(O.kl_diag(mu1, o1, mu2, o2) > -1e-9))
    gap = np.abs(O.kl_diag(mu1, o1, mu2, o2) - O.kl_diag(mu2, o2, mu1, o1))
    check("KL 非对称", np.all(gap > 1e-6), f"min|KL(P||Q)-KL(Q||P)|={gap.min():.3e}")
    bc = O.bc_diag(mu1, o1, mu2, o2)
    check("BC ∈ (0,1]", np.all((bc > 0) & (bc <= 1 + 1e-9)),
          f"range=[{bc.min():.4f},{bc.max():.4f}]")

    d_kl = np.max(np.abs(O.kl_diag(mu1, o1, mu2, o2) - kl_full(mu1, o1, mu2, o2)))
    d_bc = np.max(np.abs(O.bc_diag(mu1, o1, mu2, o2) - bc_full(mu1, o1, mu2, o2)))
    check("对角闭式 KL == 全矩阵", d_kl < 1e-9, f"max|Δ|={d_kl:.2e}")
    check("对角闭式 BC == 全矩阵", d_bc < 1e-9, f"max|Δ|={d_bc:.2e}")

    # ---------- 结构正确的三元组：parent 比 child 宽、child 贴近 parent、neg 远离 ----------
    mu_p = RNG.normal(0, 0.4, (B, D))
    mu_c = mu_p + RNG.normal(0, 0.05, (B, D))
    mu_n = mu_p + RNG.normal(0, 1.8, (B, D))
    o_p = np.exp(RNG.normal(0.55, 0.08, (B, D)))   # 宽
    o_c = np.exp(RNG.normal(-0.45, 0.08, (B, D)))  # 窄
    o_n = np.exp(RNG.normal(0.20, 0.10, (B, D)))

    lp = O.loss_paper(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    lo = O.loss_official(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    print("\n-- 论文式 7-13 --")
    for k, v in lp.items():
        print(f"   {k:11s} = {v:.6f}")
    print("-- 官方源码 forward --")
    for k, v in lo.items():
        print(f"   {k:11s} = {v:.6f}")

    check("论文损失全有限", all(np.isfinite(v) for v in lp.values()))
    check("源码损失全有限", all(np.isfinite(v) for v in lo.values()))

    # 打乱：把 child 与 negative parent 的角色互换，损失应上升
    lp_bad = O.loss_paper(mu_c, o_c, mu_n, o_n, mu_p, o_p)
    lo_bad = O.loss_official(mu_c, o_c, mu_n, o_n, mu_p, o_p)
    check("论文损失: 正例 < 打乱", lp["total"] < lp_bad["total"],
          f"{lp['total']:.4f} < {lp_bad['total']:.4f}")
    check("源码损失: 正例 < 打乱", lo["total"] < lo_bad["total"],
          f"{lo['total']:.4f} < {lo_bad['total']:.4f}")
    check("两套损失数值不等（口径确实不同）", abs(lp["total"] - lo["total"]) > 1e-6,
          f"|Δ|={abs(lp['total']-lo['total']):.4f}")

    # ---------- 数值稳定性 ----------
    o_small = np.full((B, D), 1e-3, dtype=np.float64)
    o_big = np.full((B, D), 10.0, dtype=np.float64)
    e1 = O.loss_paper(mu_c, o_small, mu_p, o_big, mu_n, o_small)
    check("极端 o 不 NaN/Inf", all(np.isfinite(v) for v in e1.values()))

    f32 = dict(mu_c=mu_c.astype(np.float32), o_c=o_c.astype(np.float32),
               mu_p=mu_p.astype(np.float32), o_p=o_p.astype(np.float32),
               mu_n=mu_n.astype(np.float32), o_n=o_n.astype(np.float32))
    e2 = O.loss_paper(**f32)
    check("float32 入参不 NaN/Inf", all(np.isfinite(v) for v in e2.values()))

    # ---------- 推理排父 ----------
    s_bc = O.rank_anchors(mu_c, o_c, np.concatenate([mu_p[:1], mu_n[:1]]),
                          np.concatenate([o_p[:1], o_n[:1]]), "BC")
    check("BC 排序把真父排在首位", s_bc[0, 0] > s_bc[0, 1], f"{s_bc[0].round(4).tolist()}")

    print(f"\n== {sum(PASS)}/{len(PASS)} PASS ==")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
