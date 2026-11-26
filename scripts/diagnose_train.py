"""L2 Jittor inf/nan 定位：一次跑完输出全部判别证据，结果写 logs/diagnose_train.json"""
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("cc_path", "/usr/bin/g++-12")

from src import model_jittor as M  # noqa: E402
from src import projection  # noqa: E402

NPZ = os.path.join(ROOT, "logs", "parity_input.npz")
OUT = os.path.join(ROOT, "logs", "diagnose_train.json")


def scalar(x):
    try:
        a = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
        return float(a.reshape(-1)[0])
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


def stat(a):
    try:
        b = np.asarray(a.numpy() if hasattr(a, "numpy") else a, dtype=np.float64)
        return dict(max_abs=float(np.abs(b).max()),
                    has_nan=bool(np.isnan(b).any()),
                    has_inf=bool(np.isinf(b).any()))
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


def main():
    report = {"npz": NPZ, "checks": {}}
    if not os.path.exists(NPZ):
        report["error"] = "缺少 npz，先跑 python scripts/parity.py --emit-input"
        dump(report)
        print(report["error"])
        return 1

    import jittor as jt
    jt.flags.use_cuda = 0

    with np.load(NPZ, allow_pickle=True) as z:
        batch_np = {k: z[k].astype(np.float32) for k in ("n_c", "n_p", "n_n")}
    batch = {k: jt.array(v) for k, v in batch_np.items()}

    w = projection.init_weights()
    model = M.TaxoBellProjection(w)
    model.train()
    params = model.param_list()
    opt = jt.optim.AdamW(params, lr=1e-3)

    # F: 参数是否可导、optimizer 收集是否一致
    try:
        report["checks"]["F_params"] = {
            "n_param_list": len(params),
            "n_parameters": len(model.parameters()),
            "stop_grad": [bool(p.is_stop_grad()) if hasattr(p, "is_stop_grad")
                          else "no_api" for p in params],
            "weight_fingerprint": projection.weight_fingerprint(w),
        }
    except Exception as e:
        report["checks"]["F_params"] = f"ERR:{type(e).__name__}:{e}"

    # A: 纯前向，不触碰 optimizer
    try:
        out = model.loss(batch, "paper")
        report["checks"]["A_forward_total"] = scalar(out["total"])
        report["checks"]["A_forward_parts"] = {
            k: scalar(v) for k, v in out.items() if k != "total"}
    except Exception as e:
        report["checks"]["A_forward_total"] = f"ERR:{type(e).__name__}:{e}"

    # B/C/D/E: backward -> 读梯度 -> step 前取值 -> step -> 参数变化 -> 重算
    try:
        out = model.loss(batch, "paper")
        before = [p.numpy().copy() for p in params]
        opt.backward(out["total"])
        report["checks"]["B_total_before_step"] = scalar(out["total"])
        grads = []
        for p in params:
            # Jittor 1.3.11.0：Var.grad 不是可访问属性，须用 opt_grad(optimizer)
            # （诊断脚本首次实跑的 E_error 原文提示）。PyTorch 的 .grad 在此不适用。
            g = p.opt_grad(opt)
            grads.append(stat(g) if g is not None else "grad_none")
        report["checks"]["C_grads"] = grads
        opt.step()
        jt.sync_all()
        report["checks"]["D_param_delta"] = [
            stat(p.numpy() - b) for p, b in zip(params, before)]
        report["checks"]["E_total_after_step_same_graph"] = scalar(out["total"])
        out2 = model.loss(batch, "paper")
        report["checks"]["E_total_after_step_new_graph"] = scalar(out2["total"])
    except Exception as e:
        report["checks"]["E_error"] = f"{type(e).__name__}:{e}\n{traceback.format_exc()}"

    # G: 用修好的 train_step 连走 5 步（全新模型 + 全新 optimizer）
    try:
        model2 = M.TaxoBellProjection(projection.init_weights())
        model2.train()
        opt2 = jt.optim.AdamW(model2.param_list(), lr=1e-3)
        curve = []
        for s in range(5):
            r = model2.train_step(batch, opt2, "paper")
            curve.append(r["total"])
        report["checks"]["G_train_step_curve"] = curve
    except Exception as e:
        report["checks"]["G_error"] = f"{type(e).__name__}:{e}\n{traceback.format_exc()}"

    dump(report)
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False, default=str))
    print(f"\n-> {OUT}")
    return 0


def dump(report):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    sys.exit(main())
