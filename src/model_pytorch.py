"""PyTorch 对照实现：投影头 → (mu, o) → 损失，供 Jittor 主体逐项对照"""
import torch

from src import ops_torch as O
from src.projection import EPS, WEIGHT_KEYS


class TaxoBellProjection(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        for k in WEIGHT_KEYS:
            setattr(self, k, torch.nn.Parameter(
                torch.tensor(w[k], dtype=torch.float32)))

    def forward(self, n):
        h = n @ self.W1 + self.b1
        h = h * (h > 0)
        mu = h @ self.Wc + self.bc
        o = torch.exp(h @ self.Wo + self.bo) + EPS
        return mu, o

    def loss(self, batch, loss_name="paper"):
        mu_c, o_c = self.forward(batch["n_c"])
        mu_p, o_p = self.forward(batch["n_p"])
        mu_n, o_n = self.forward(batch["n_n"])
        fn = O.loss_paper if loss_name == "paper" else O.loss_official
        return fn(mu_c, o_c, mu_p, o_p, mu_n, o_n)

    def train_step(self, batch, opt, loss_name="paper"):
        opt.zero_grad()
        out = self.loss(batch, loss_name)
        out["total"].backward()
        opt.step()
        return {k: float(v.detach()) for k, v in out.items()}
