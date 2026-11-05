"""VGGT 稠密深度先验的 Pearson 逆深度损失（VGGT-Splat 新增，不来自 gsplat）。

- 只约束“形状”：渲染逆深度与先验逆深度各自在有效像素上去均值、除标准差后的相关系数，
  loss = 1 - rho ∈ [0, 2]。先验乘任意正尺度（深度单位、世界归一化、pose opt）都不改变损失
- 逆深度而不是深度：深度下远处像素偏离均值最大、主导协方差，而远处正是先验最不可靠、对 3DGS 最不要紧的地方
- 有效像素 = 先验置信度够高（掩码由调用方给）且渲染累积不透明度 > alpha_thres
  （expected depth 已除以累积不透明度，稀疏处数值不稳定）且两边深度为正
- 分辨率：渲染深度 / 不透明度用 area 插值缩到先验分辨率（VGGT 模型分辨率，如 518×294）再比，不放大先验
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def pearson_loss(x: Tensor, y: Tensor, mask: Tensor, eps: float = 1e-6, min_pixels: int = 16) -> Tensor:
    """逐图 1 - Pearson(x, y)，只用 mask 内的像素，对有效图取平均。

    x, y: (B, H, W)；mask: (B, H, W) bool。有效像素少于 min_pixels 的图跳过；全部跳过时返回 0（仍可反传）。
    """
    losses = []
    for xi, yi, mi in zip(x, y, mask):
        if mi.sum() < min_pixels:
            continue
        a, b = xi[mi], yi[mi]
        # 先除以平均幅值（Pearson 对正尺度不变）：否则先验单位很大时逆深度 ~1e-5，eps 不再可忽略
        a = a / (a.abs().mean().detach() + 1e-12)
        b = b / (b.abs().mean().detach() + 1e-12)
        a = a - a.mean()
        b = b - b.mean()
        rho = (a * b).sum() / (a.norm() * b.norm() + eps)
        losses.append(1.0 - rho)
    if not losses:
        return x.sum() * 0.0
    return torch.stack(losses).mean()


def dense_depth_loss(
    render_depth: Tensor,
    render_alpha: Tensor,
    prior_depth: Tensor,
    prior_mask: Tensor,
    alpha_thres: float = 0.5,
    min_depth: float = 1e-6,
    eps: float = 1e-6,
) -> Tensor:
    """渲染深度与先验深度的 Pearson 逆深度损失。

    render_depth, render_alpha: (B, H, W, 1)（gsplat RGB+ED 的深度通道与 alphas，训练分辨率）
    prior_depth: (B, h, w) 先验深度（VGGT，模型分辨率）；prior_mask: (B, h, w) bool，先验置信度够高的像素
    """
    h, w = prior_depth.shape[-2:]
    d = render_depth.permute(0, 3, 1, 2)
    a = render_alpha.permute(0, 3, 1, 2)
    if d.shape[-2:] != (h, w):
        d = F.interpolate(d, size=(h, w), mode="area")
        a = F.interpolate(a, size=(h, w), mode="area")
    d, a = d[:, 0], a[:, 0]
    valid = prior_mask & (a > alpha_thres) & (d > min_depth) & (prior_depth > min_depth)
    inv_render = 1.0 / d.clamp(min=min_depth)
    inv_prior = 1.0 / prior_depth.clamp(min=min_depth)
    return pearson_loss(inv_render, inv_prior, valid, eps=eps)


def confidence_mask(conf: Tensor, depth: Tensor, percentile: Optional[float]) -> Tensor:
    """按百分位取置信度阈值（严格大于，与 vggt_to_colmap.py --conf_percentile 一致：
    desk 上过半像素 conf 恰好是 1.0，>= 会把它们全部放进来）。percentile=None 时只要求深度为正。

    阈值在全部帧的有效像素上统一算。conf, depth: (S, h, w)。
    """
    valid = depth > 0
    if percentile is None:
        return valid
    # numpy 而不是 torch.quantile：后者输入上限 2^24 个元素，180 帧 × 518×294 会超
    thres = float(np.percentile(conf[valid].cpu().numpy(), percentile))
    return valid & (conf > thres)
