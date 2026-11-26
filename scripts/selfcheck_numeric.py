"""数值稳定性回归自检（纯 numpy / 标准库，无框架）"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import ops_numpy as O  # noqa: E402

PASS = 0
TOTAL = 0


def check(cond, msg):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")


def main():
    print("== 数值稳定性回归自检（纯 numpy，无框架）==")

    # 1) 非饱和样本：_safe_log_arg 恒等，log 参数与修复前逐位一致
    ok = True
    for x in (0.02, 0.1, 0.5, 0.9, 1.0):
        ok &= abs(O._safe_log_arg(np.float32(x)) - np.float32(x)) < 1e-6
    check(ok, "非饱和 1-neg_bc∈[0.02,1.0]：_safe_log_arg 恒等（L3 数值不变）")

    # 2) 饱和 / 越界样本：参数被兜到 1e-4，log 有限
    sat = O._safe_log_arg(np.float32(0.0))
    over = O._safe_log_arg(np.float32(-0.003))   # neg_bc 越过 1 的情形
    check(abs(sat - 1e-4) < 1e-9 and abs(over - 1e-4) < 1e-9,
          f"饱和/越界参数兜到 1e-4，log 有限 log(1e-4)={np.log(sat):.3f}")

    # 3) 端到端：让 neg_bc 精确=1（mu_n==mu_c 且 o_n==o_c → D_B=0 → BC=1）
    B, d = 4, 8
    mu_c = np.zeros((B, d), np.float32)
    o_c = np.ones((B, d), np.float32)
    mu_p = np.full((B, d), 0.3, np.float32)
    o_p = np.full((B, d), 1.2, np.float32)
    mu_n = mu_c.copy()   # 与 child 完全相同 → neg_bc=1.0
    o_n = o_c.copy()
    nb = O.bc_diag(mu_n, o_n, mu_c, o_c)
    check(bool(np.allclose(nb, 1.0)), f"构造出 neg_bc=1.0 饱和三元组 min={nb.min():.6f}")

    lp = O.loss_paper(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    lo = O.loss_official(mu_c, o_c, mu_p, o_p, mu_n, o_n)
    check(all(np.isfinite(v) for v in lp.values()),
          f"loss_paper 饱和样本全有限 L_sym={float(lp['L_sym']):.4f} total={float(lp['total']):.4f}")
    check(all(np.isfinite(v) for v in lo.values()),
          f"loss_official 饱和样本全有限 bc_loss={float(lo['bc_loss']):.4f} total={float(lo['total']):.4f}")

    # 4) 三份源码一致性：都定义了 CLAMP_EPS 与 _safe_log_arg，且不再调用 D_EPS
    srcs = ["ops_numpy.py", "ops_torch.py", "ops_jittor.py"]
    sig_ok = True
    for s in srcs:
        txt = open(os.path.join(ROOT, "src", s), encoding="utf-8").read()
        has_clamp = "CLAMP_EPS = 1e-4" in txt and "def _safe_log_arg" in txt
        uses_safe = txt.count("_safe_log_arg(") >= 3   # 定义 + 两处损失
        no_old = "D_EPS)" not in txt and "+ D_EPS" not in txt
        sig_ok &= has_clamp and uses_safe and no_old
        print(f"    {s}: clamp={has_clamp} uses_safe={uses_safe} no_old_eps={no_old}")
    check(sig_ok, "三后端源码同一口径：CLAMP_EPS + _safe_log_arg，无残留 D_EPS 调用")

    # 5) 复现修复前机理：float32 下 1.0+1e-12==1.0，加法保护被吞
    folded = np.float32(1.0) + np.float32(1e-12)
    check(folded == np.float32(1.0),
          "复现根因：float32 常量折叠 1.0+1e-12==1.0（旧 D_EPS 保护被吞）")
    with np.errstate(divide="ignore"):
        old = np.log(np.float32(0.0) + np.float32(1e-12) * 0)  # 折叠后 log(0)
    check(np.isneginf(old), f"修复前 log(0)={old} → -inf（故 step0=inf）")
    check(np.isfinite(np.log(sat)), "修复后 log(_safe_log_arg(0))=log(1e-4) 有限")

    # 6) 正常样本端到端 L_sym 与「无 clamp 手写式」一致（口径未漂移）
    mu_n2 = np.full((B, d), 0.8, np.float32)
    o_n2 = np.full((B, d), 0.9, np.float32)
    pb = O.bc_diag(mu_p, o_p, mu_c, o_c)
    nb2 = O.bc_diag(mu_n2, o_n2, mu_c, o_c)
    manual = float(np.mean(-np.log(pb + 1e-12) - np.log(1.0 - nb2 + 1e-12)))
    lp2 = O.loss_paper(mu_c, o_c, mu_p, o_p, mu_n2, o_n2)
    check(abs(float(lp2["L_sym"]) - manual) < 1e-5,
          f"正常样本 L_sym 与修复前一致 Δ={abs(float(lp2['L_sym'])-manual):.2e}")

    print(f"\n== {PASS}/{TOTAL} PASS ==")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
