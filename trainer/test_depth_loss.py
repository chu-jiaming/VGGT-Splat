"""depth_loss.py 的单元测试（CPU 即可）：python trainer/test_depth_loss.py 或 pytest trainer/test_depth_loss.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_loss import confidence_mask, dense_depth_loss, pearson_loss  # noqa: E402

torch.manual_seed(0)
B, H, W = 2, 54, 96      # 渲染（训练）分辨率
h, w = 27, 48            # 先验分辨率


def scene_depth():
    """平滑但有结构的正深度图 (B, h, w)。"""
    y = torch.linspace(0, 1, h)[:, None]
    x = torch.linspace(0, 1, w)[None, :]
    base = 2.0 + 3.0 * y + torch.sin(6 * x) * 0.5
    return torch.stack([base, base.flip(-1) * 1.3])


def upsample(d):
    """先验分辨率 → 渲染分辨率 (B, H, W, 1)，area 缩回去后与原图一致（整数倍最近邻）。"""
    return d.repeat_interleave(H // h, 1).repeat_interleave(W // w, 2)[..., None]


def full_alpha():
    return torch.ones(B, H, W, 1)


def all_valid():
    return torch.ones(B, h, w, dtype=torch.bool)


def test_perfect_match_is_zero():
    d = scene_depth()
    loss = dense_depth_loss(upsample(d), full_alpha(), d, all_valid())
    assert loss.abs() < 1e-5, loss


def test_prior_positive_scale_invariant():
    """roadmap L5 的产出要求：先验乘任意正尺度，损失不变。"""
    d = scene_depth()
    render = upsample(d + 0.3 * torch.randn_like(d).abs())
    prior = scene_depth()
    ref = dense_depth_loss(render, full_alpha(), prior, all_valid())
    for s in (1e-3, 0.5, 7.0, 1e4):
        loss = dense_depth_loss(render, full_alpha(), prior * s, all_valid())
        assert torch.allclose(loss, ref, atol=1e-5), (s, loss, ref)
    assert ref > 1e-4  # 不是恒为 0


def test_pearson_affine_invariant_and_sign():
    x = torch.rand(1, h, w) + 0.1
    m = all_valid()[:1]
    assert pearson_loss(x, 3 * x + 2, m) < 1e-5
    assert torch.allclose(pearson_loss(x, -x, m), torch.tensor(2.0), atol=1e-5)


def test_masked_pixels_ignored():
    d = scene_depth()
    prior = d.clone()
    m = all_valid()
    m[:, :, : w // 2] = False
    prior[:, :, : w // 2] = 100.0 * torch.rand(B, h, w // 2) + 0.1  # 掩码外的先验全是噪声
    loss = dense_depth_loss(upsample(d), full_alpha(), prior, m)
    assert loss.abs() < 1e-5, loss


def test_low_alpha_ignored():
    d = scene_depth()
    render = upsample(d).clone()
    alpha = full_alpha()
    render[:, :, : W // 2] = 50.0  # 稀疏处的离谱深度
    alpha[:, :, : W // 2] = 0.1
    loss = dense_depth_loss(render, alpha, d, all_valid(), alpha_thres=0.5)
    assert loss.abs() < 1e-5, loss


def test_gradient_flows_and_finite():
    d = scene_depth()
    render = upsample(d + 0.5).requires_grad_(True)
    loss = dense_depth_loss(render, full_alpha(), d * 2, all_valid())
    loss.backward()
    assert torch.isfinite(render.grad).all()
    # 常数深度（标准差 0）也不能出 NaN
    flat = torch.full((B, H, W, 1), 3.0, requires_grad=True)
    loss = dense_depth_loss(flat, full_alpha(), d, all_valid())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flat.grad).all()


def test_empty_mask_returns_zero_with_grad():
    render = torch.rand(B, H, W, 1, requires_grad=True)
    loss = dense_depth_loss(render, full_alpha(), scene_depth(), torch.zeros(B, h, w, dtype=torch.bool))
    loss.backward()
    assert loss == 0


def test_confidence_mask_strict_percentile():
    conf = torch.ones(2, h, w)
    conf[0, : h // 4] = 5.0  # 少量高置信度，其余恰好 1.0（desk 的情况）
    depth = torch.ones(2, h, w)
    m = confidence_mask(conf, depth, 50)
    assert m.sum() == (conf > 1.0).sum()  # 严格大于：conf == 1.0 的全部排除
    assert confidence_mask(conf, depth, None).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
