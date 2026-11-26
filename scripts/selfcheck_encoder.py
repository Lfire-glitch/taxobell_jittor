"""编码器层契约自检：纯 numpy / 标准库，不触发 torch / jittor / transformers"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import encoder_jittor as E  # noqa: E402

NPZ = os.path.join(ROOT, "logs", "parity_input.npz")
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


def main():
    print("== 编码器层契约自检（纯 numpy，无框架）==")
    check("encoder_jittor 可 import（框架延迟加载）", hasattr(E, "encode_jittor"))
    check("默认旧仓库路径写死为 WSL 原生路径",
          E.DEFAULT_REPO.startswith("/home/guo76/"), E.DEFAULT_REPO)

    if not os.path.exists(NPZ):
        print(f"  跳过 cached 检查：{NPZ} 不存在，先跑 parity.py --emit-input")
    else:
        c = E.cached_embeddings(NPZ)
        with np.load(NPZ, allow_pickle=True) as z:
            same = all(np.array_equal(c[k], z[k].astype(np.float32))
                       for k in ("n_c", "n_p", "n_n"))
        check("cached n_* 与 npz 逐元素一致", same,
              f"shapes={[c[k].shape for k in c]}")

    a = np.random.default_rng(7).normal(0, 1, (4, 8)).astype(np.float32)
    r_id = E.cls_parity(a, a)
    check("完全相同 [CLS]：cos=1，max|Δ|=0",
          abs(r_id["cos_mean"] - 1.0) < 1e-9 and r_id["max_abs_delta"] == 0.0,
          f"cos={r_id['cos_mean']:.8f} maxΔ={r_id['max_abs_delta']:.1e}")
    b = a + 1e-2
    r_near = E.cls_parity(a, b)
    check("轻微扰动：cos 下降且 max|Δ| 为正",
          r_near["cos_mean"] < r_id["cos_mean"] and r_near["max_abs_delta"] > 0,
          f"cos={r_near['cos_mean']:.6f} maxΔ={r_near['max_abs_delta']:.2e}")
    check("cls_parity 返回 shape 字段", r_id["shape"] == [4, 8], str(r_id["shape"]))

    print(f"\n== {sum(PASS)}/{len(PASS)} PASS ==")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
