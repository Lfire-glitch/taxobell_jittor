"""Jittor 主体实现：投影头 → (mu, o) → 损失 → 训练 step（主线训练与前向归 Jittor）"""
import os

os.environ.setdefault("cc_path", "/usr/bin/g++-12")  # 必须在 import jittor 之前
import numpy as np  # noqa: E402
import jittor as jt  # noqa: E402

jt.flags.use_cuda = 0  # 全 CPU，与旧仓库一致

from src import ops_jittor as O  # noqa: E402
from src.projection import EPS, WEIGHT_KEYS  # noqa: E402


class TaxoBellProjection(jt.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.params = {}
        for k in WEIGHT_KEYS:
            v = jt.array(w[k])
            setattr(self, k, v)
            self.params[k] = v

    def param_list(self):
        """显式参数列表：不赌 setattr 的裸 jt.array 会不会被 parameters() 收集。"""
        return [self.params[k] for k in WEIGHT_KEYS]

    def execute(self, n):
        h = n @ self.W1 + self.b1
        h = h * (h > 0)
        mu = h @ self.Wc + self.bc
        o = O._exp(h @ self.Wo + self.bo) + EPS
        return mu, o

    def loss(self, batch, loss_name="paper"):
        mu_c, o_c = self.execute(batch["n_c"])
        mu_p, o_p = self.execute(batch["n_p"])
        mu_n, o_n = self.execute(batch["n_n"])
        fn = O.loss_paper if loss_name == "paper" else O.loss_official
        return fn(mu_c, o_c, mu_p, o_p, mu_n, o_n)

    def train_step(self, batch, opt, loss_name="paper"):
        out = self.loss(batch, loss_name)
        opt.backward(out["total"])          # 反传同时把整张图执行、中间 Var 物化
        rec = {k: float(v.item()) for k, v in out.items()}  # 必须在 step() 之前取值
        opt.step()                         # 原地更新参数，会污染上面那张图
        jt.sync_all()                      # 旧仓库实测范式：确保更新落定再进下一步
        return rec
