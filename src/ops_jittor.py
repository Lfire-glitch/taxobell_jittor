"""Jittor 实现：与 ops_numpy.py / ops_torch.py 逐字对齐（L2 训练等价的目标）"""
import jittor as jt

EPS = 1e-6
CLAMP_EPS = 1e-4    # 与 ops_numpy.py / ops_torch.py 逐字对齐：float32 安全的 log 下界


# ---------- 4 个 API shim：WSL 实跑已验证 ----------
def _sum_last(x):
    return x.sum(-1)


def _log(x):
    return jt.log(x)


def _exp(x):
    return jt.exp(x)


def _relu(x):
    return x * (x > 0)


# -----------------------------------------------------------
def _v(o):
    return (o + EPS) ** 2


def _safe_log_arg(x):
    """max(x, CLAMP_EPS)，用 relu 表达，避开 clamp API 差异与常量折叠吞 eps。

    修复前 -log(1-neg_bc+1e-12) 在 neg_bc→1 时 log(0)=-inf（float32 下
    1.0+1e-12==1.0，加法保护被优化掉）。下界 1e-4 可表示、抗融合，
    对正常样本 1-neg_bc >> 1e-4 数值完全不变。
    """
    return _relu(x - CLAMP_EPS) + CLAMP_EPS


def logdet_diag(o):
    return 2.0 * _sum_last(_log(o + EPS))


def log_volume(o):
    return _sum_last(_log(o + EPS))


def kl_diag(mu1, o1, mu2, o2):
    v1, v2 = _v(o1), _v(o2)
    d = mu1.shape[-1]
    trace = _sum_last(v1 / v2)
    mahal = _sum_last((mu2 - mu1) ** 2 / v2)
    return 0.5 * (trace + mahal - d + logdet_diag(o2) - logdet_diag(o1))


def bc_distance_diag(mu1, o1, mu2, o2):
    v1, v2 = _v(o1), _v(o2)
    m = 0.5 * (v1 + v2)
    term1 = 0.125 * _sum_last((mu1 - mu2) ** 2 / m)
    term2 = 0.5 * (_sum_last(_log(m)) - _sum_last(_log(o1 + EPS))
                   - _sum_last(_log(o2 + EPS)))
    return term1 + term2


def bc_diag(mu1, o1, mu2, o2):
    return _exp(-bc_distance_diag(mu1, o1, mu2, o2))


def _reg_paper(o, delta_var, M_var):
    v = _v(o)
    d = o.shape[-1]
    L_reg = (_relu(delta_var - v) ** 2).sum(-1).mean() / d
    L_clip = _relu(v - M_var).sum(-1).mean() / d
    return L_reg, L_clip


def loss_paper(mu_c, o_c, mu_p, o_p, mu_n, o_n,
               margin=0.5, lam=0.3, C=1.5, delta_var=0.5, M_var=10.0):
    pos_bc = bc_diag(mu_p, o_p, mu_c, o_c)
    neg_bc = bc_diag(mu_n, o_n, mu_c, o_c)
    L_sym = (-_log(_safe_log_arg(pos_bc))
             - _log(_safe_log_arg(1.0 - neg_bc))).mean()

    pos_kl = kl_diag(mu_c, o_c, mu_p, o_p)
    neg_kl = kl_diag(mu_c, o_c, mu_n, o_n)
    L_align = _relu(pos_kl - neg_kl + margin).mean()

    D_rep = log_volume(o_p) - log_volume(o_c)
    rev_kl = kl_diag(mu_p, o_p, mu_c, o_c)
    L_diverge = _relu(C * D_rep - rev_kl).mean()
    L_asym = L_align + lam * L_diverge

    regs = [_reg_paper(o, delta_var, M_var) for o in (o_p, o_c, o_n)]
    L_reg = sum(r[0] for r in regs) / 3.0
    L_clip = sum(r[1] for r in regs) / 3.0
    total = L_sym + L_asym + L_reg + L_clip
    return dict(L_sym=L_sym, L_align=L_align, L_diverge=L_diverge,
                L_asym=L_asym, L_reg=L_reg, L_clip=L_clip, total=total)


def loss_official(mu_c, o_c, mu_p, o_p, mu_n, o_n,
                  margin=0.5, lam=0.1, C=1.5,
                  wtbce=0.45, wtkl=0.45, wtreg=0.1,
                  vol_threshold=0.5, clip_min=1.0, clip_max=10.0):
    pos_bc = bc_diag(mu_p, o_p, mu_c, o_c)
    neg_bc = bc_diag(mu_n, o_n, mu_c, o_c)
    bc_loss = (-_log(_safe_log_arg(pos_bc))
               - _log(_safe_log_arg(1.0 - neg_bc))).mean()

    pos_kld = kl_diag(mu_c, o_c, mu_p, o_p)
    neg_kld = kl_diag(mu_c, o_c, mu_n, o_n)
    pos_kld_other = kl_diag(mu_p, o_p, mu_c, o_c)
    dynamic = C * (logdet_diag(o_p) - logdet_diag(o_c))
    kl_containment = dynamic - pos_kld_other
    kl_loss = _relu(pos_kld + margin - neg_kld + lam * kl_containment).mean()

    def reg(o):
        std = o + EPS
        minvol = (_relu(vol_threshold - std) ** 2).mean()
        v = _v(o)
        clip = (_relu(clip_min - v) ** 2 + _relu(v - clip_max) ** 2).mean()
        return minvol + clip

    reg_sum = reg(o_p) + reg(o_c) + reg(o_n)
    total = wtbce * bc_loss + wtkl * kl_loss + wtreg * reg_sum
    return dict(bc_loss=bc_loss, kl_loss=kl_loss, reg_sum=reg_sum, total=total)
