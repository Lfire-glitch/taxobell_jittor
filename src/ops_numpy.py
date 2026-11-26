"""TaxoBell §3/§4 闭式算子的纯 numpy 参考实现（对角高斯）"""
import numpy as np

EPS = 1e-6          # 对齐源码 BC 里 1e-6*eye 的量级
CLAMP_EPS = 1e-4    # log 参数下界；须远大于 float32 在 1.0 附近的 eps(1.2e-7)


def _v(o):
    """diag(Σ) 的元素，恒正。"""
    return (o + EPS) ** 2


def _relu(x):
    return x * (x > 0)


def _safe_log_arg(x):
    """max(x, CLAMP_EPS)，用 relu 表达以避开各后端 clamp API 差异。

    -log(1-neg_bc) 在 neg_bc→1 时参数塌到 0；原式加 1e-12 会被 Jittor 常量折叠
    成 1.0-neg_bc 而丢失保护。下界取 1e-4 可表示且抗融合，对正常样本
    (1-neg_bc >> 1e-4) 数值完全不变，故 L3 parity 仍逐位一致。
    """
    return _relu(x - CLAMP_EPS) + CLAMP_EPS


def logdet_diag(o):
    """log det Σ = 2 Σ_i log(o_i + EPS)。"""
    return 2.0 * np.sum(np.log(o + EPS), axis=-1)


def log_volume(o):
    """论文 §4.3 的可微体积代理：1/2 log det Σ = Σ_i log(o_i + EPS)。

    注意：源码 log_det_term() 返回的是完整的 log det Σ（= 2 × 本函数），
    因此论文 L_diverge 与源码 dynamic 相差一个因子 2。
    """
    return np.sum(np.log(o + EPS), axis=-1)


def kl_diag(mu1, o1, mu2, o2):
    """D_KL(N1 || N2)，非对称。式 4 的对角闭式：

    1/2 [ tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - d + log(detΣ2/detΣ1) ]

    tr(Σ2^-1 Σ1) = Σ_i v1_i/v2_i ；Mahalanobis = Σ_i (μ2_i-μ1_i)^2 / v2_i
    """
    v1, v2 = _v(o1), _v(o2)
    d = mu1.shape[-1]
    trace = np.sum(v1 / v2, axis=-1)
    mahal = np.sum((mu2 - mu1) ** 2 / v2, axis=-1)
    return 0.5 * (trace + mahal - d + logdet_diag(o2) - logdet_diag(o1))


def bc_distance_diag(mu1, o1, mu2, o2):
    """D_B = 1/8 Δμ^T Σm^-1 Δμ + 1/2 log( detΣm / sqrt(detΣ1 detΣ2) )，Σm=(Σ1+Σ2)/2。"""
    v1, v2 = _v(o1), _v(o2)
    m = 0.5 * (v1 + v2)
    term1 = 0.125 * np.sum((mu1 - mu2) ** 2 / m, axis=-1)
    # log det Σm = Σ log m_i ；1/2 log det Σ1 = Σ log(o1_i+EPS)
    term2 = 0.5 * (np.sum(np.log(m), axis=-1)
                   - np.sum(np.log(o1 + EPS), axis=-1)
                   - np.sum(np.log(o2 + EPS), axis=-1))
    return term1 + term2


def bc_diag(mu1, o1, mu2, o2):
    """BC = exp(-D_B) ∈ (0, 1]，1 表示两个高斯完全重合。"""
    return np.exp(-bc_distance_diag(mu1, o1, mu2, o2))


def _reg_paper(o, delta_var, M_var):
    v = _v(o)
    d = o.shape[-1]
    L_reg = np.mean(np.sum(_relu(delta_var - v) ** 2, axis=-1) / d)
    L_clip = np.mean(np.sum(_relu(v - M_var), axis=-1) / d)
    return L_reg, L_clip


def loss_paper(mu_c, o_c, mu_p, o_p, mu_n, o_n,
               margin=0.5, lam=0.3, C=1.5, delta_var=0.5, M_var=10.0):
    """论文式 7–13：L_overall = L_sym + L_asym + L_reg + L_clip（无权重直接相加）。

    一个训练实例 = 1 正 + 1 负三元组 (child, parent, negative_parent)。
    N 个 hard negatives 由数据层展开 / 求平均，算子层只处理三元组（与源码 forward 一致）。
    各分量对 batch 取 mean，返回标量 dict。
    """
    pos_bc = bc_diag(mu_p, o_p, mu_c, o_c)
    neg_bc = bc_diag(mu_n, o_n, mu_c, o_c)
    L_sym = np.mean(-np.log(_safe_log_arg(pos_bc))
                    - np.log(_safe_log_arg(1.0 - neg_bc)))

    pos_kl = kl_diag(mu_c, o_c, mu_p, o_p)
    neg_kl = kl_diag(mu_c, o_c, mu_n, o_n)
    L_align = np.mean(_relu(pos_kl - neg_kl + margin))

    D_rep = log_volume(o_p) - log_volume(o_c)          # 1/2 log det 之差
    rev_kl = kl_diag(mu_p, o_p, mu_c, o_c)             # D_KL(N_p || N_c)
    L_diverge = np.mean(_relu(C * D_rep - rev_kl))
    L_asym = L_align + lam * L_diverge

    regs = [_reg_paper(o, delta_var, M_var) for o in (o_p, o_c, o_n)]
    L_reg = np.mean([r[0] for r in regs])
    L_clip = np.mean([r[1] for r in regs])

    total = L_sym + L_asym + L_reg + L_clip
    return dict(L_sym=L_sym, L_align=L_align, L_diverge=L_diverge,
                L_asym=L_asym, L_reg=L_reg, L_clip=L_clip, total=total)


def loss_official(mu_c, o_c, mu_p, o_p, mu_n, o_n,
                  margin=0.5, lam=0.1, C=1.5,
                  wtbce=0.45, wtkl=0.45, wtreg=0.1,
                  vol_threshold=0.5, clip_min=1.0, clip_max=10.0):
    """官方源码 model.py::forward 的可执行复刻（exp.py::train_one_step 实际调用这套）。

    与论文的三处实质差异：
      1) L_align 与 L_diverge 合并进单个 hinge，且 λ 在 hinge 内部乘 kl_containment；
      2) dynamic 用完整 log det Σ（不是论文的 1/2 log det），故反向 KL 项尺度 ×2；
      3) 总损失带 0.45/0.45/0.1 权重，reg = min_volume(std) + clip_covariance(var)，
         对 parent/child/negative 三者分别做（std 与 variance 混用）。
    """
    pos_bc = bc_diag(mu_p, o_p, mu_c, o_c)
    neg_bc = bc_diag(mu_n, o_n, mu_c, o_c)
    bc_loss = np.mean(-np.log(_safe_log_arg(pos_bc))
                      - np.log(_safe_log_arg(1.0 - neg_bc)))

    pos_kld = kl_diag(mu_c, o_c, mu_p, o_p)
    neg_kld = kl_diag(mu_c, o_c, mu_n, o_n)
    pos_kld_other = kl_diag(mu_p, o_p, mu_c, o_c)      # D_KL(N_p || N_c)
    dynamic = C * (logdet_diag(o_p) - logdet_diag(o_c))  # 完整 log det
    kl_containment = dynamic - pos_kld_other
    kl_loss = np.mean(_relu(pos_kld + margin - neg_kld + lam * kl_containment))

    def reg(o):
        std = o + EPS
        minvol = np.mean(_relu(vol_threshold - std) ** 2)   # 源码对 std
        v = _v(o)
        clip = np.mean(_relu(clip_min - v) ** 2 + _relu(v - clip_max) ** 2)  # 源码对 variance
        return minvol + clip

    reg_sum = reg(o_p) + reg(o_c) + reg(o_n)
    total = wtbce * bc_loss + wtkl * kl_loss + wtreg * reg_sum
    return dict(bc_loss=bc_loss, kl_loss=kl_loss, reg_sum=reg_sum, total=total)


def rank_anchors(mu_q, o_q, mu_a, o_a, mode="BC"):
    """推理：对每个 query 在所有候选 anchor 上打分，返回 [Q, A] 分数（越大越好）。

    mode="BC" → TaxoBell_BC（对称重叠）；mode="KL" → TaxoBell_KL（-D_KL 排序）。
    """
    q_mu, q_o = mu_q[:, None, :], o_q[:, None, :]
    a_mu, a_o = mu_a[None, :, :], o_a[None, :, :]
    if mode == "BC":
        return bc_diag(q_mu, q_o, a_mu, a_o)
    if mode == "KL":
        return -kl_diag(q_mu, q_o, a_mu, a_o)
    raise ValueError("mode must be 'BC' or 'KL'")
