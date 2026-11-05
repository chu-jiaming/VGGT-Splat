"""诊断：训练好的 3DGS 渲染深度和 VGGT 深度先验有多一致（D10，判断深度先验是否多余）。

vggt 方法的初始点云就是 VGGT 深度在置信度 > p50 像素上的反投影，和 --dense_depth 损失用的是同一批像素。
如果不加深度损失的 vggt 运行训练完，渲染深度和先验的相关已经很高，深度损失就没有新信息可加。

对每个运行，在训练帧（pose opt 之后的位姿、训练内参）渲染 RGB+ED，缩到先验分辨率后算逆深度 Pearson：
    rho_loss     损失掩码（conf > p50 且 alpha > 0.5）上的整图相关，即 1 - 训练时的 dense depth loss
    rho_lowconf  低置信度像素（conf ≤ p50 且 alpha > 0.5）：损失管不到、初始化也没点的区域
    rho_patch    损失掩码上 32×32 分块各算一个相关再平均：整图相关看不出的局部偏差
    alpha_cov    alpha > 0.5 的像素比例
每个运行最多取 --max_frames 帧（均匀抽样）。

用法（服务器）：
    ~/work/.venv/bin/python scripts/depth_diag.py --runs /root/autodl-tmp/runs \\
        --pairs "desk/vggt_8 desk/vggt_depth_w0.2_8 ..." --out /root/autodl-tmp/results/depth_diag.csv
不给 --pairs 时：每个场景、每个 N 的 vggt_<N> 和 vggt_depth_w<λ>_<N>（--lambda，默认 0.2）
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import Parser, load_cfg, load_gaussians, load_target, latest_ckpt, rasterization  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer"))
from depth_loss import confidence_mask  # noqa: E402

SCENES = ["desk", "lawn", "room", "garden", "bicycle", "counter"]
NS = ["8", "16", "32", "all"]


def render_depth(gs, c2w, K, W, H, cfg):
    """→ 深度 (H, W)、alpha (H, W)；RGB+ED 的深度已除以累积不透明度。"""
    out, alphas, _ = rasterization(
        means=gs["means"], quats=gs["quats"], scales=gs["scales"],
        opacities=gs["opacities"], colors=gs["colors"],
        viewmats=torch.linalg.inv(c2w)[None], Ks=K[None], width=W, height=H,
        sh_degree=cfg.get("sh_degree", 3),
        near_plane=cfg.get("near_plane", 0.01), far_plane=cfg.get("far_plane", 1e10),
        packed=cfg.get("packed", False),
        rasterize_mode="antialiased" if cfg.get("antialiased") else "classic",
        camera_model="pinhole", render_mode="RGB+ED",
    )
    return out[0, ..., 3], alphas[0, ..., 0]


def pearson(x, y):
    x = x / x.abs().mean()
    y = y / y.abs().mean()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


def patch_pearson(x, y, m, size=32, min_px=64):
    vals = []
    h, w = x.shape
    for i in range(0, h - size + 1, size):
        for j in range(0, w - size + 1, size):
            mm = m[i:i + size, j:j + size]
            if mm.sum() >= min_px:
                vals.append(pearson(x[i:i + size, j:j + size][mm], y[i:i + size, j:j + size][mm]))
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def diag_run(run_dir, max_frames, device):
    gs_dir = os.path.join(run_dir, "gs")
    cfg = load_cfg(gs_dir)
    ckpt = torch.load(latest_ckpt(gs_dir), map_location="cpu", weights_only=False)
    gs = load_gaussians(ckpt, device)
    c2ws, _, Ks = load_target(cfg, ckpt)
    p = Parser(cfg["data_dir"], factor=cfg["data_factor"], normalize=cfg.get("normalize_world_space", True), test_every=0)
    size = {n: p.imsize_dict[p.camera_ids[i]] for i, n in enumerate(p.image_names)}

    pred = np.load(os.path.join(cfg["data_dir"], "vggt_pred.npz"))
    names = [str(n) for n in pred["names"]]
    depth = torch.from_numpy(pred["depth"]).float()
    conf = torch.from_numpy(pred["conf"]).float()
    hi = confidence_mask(conf, depth, cfg.get("dense_depth_conf_percentile") or 50.0)
    h, w = depth.shape[-2:]

    frames = [n for n in p.image_names if n in names]
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    acc = {k: [] for k in ("rho_loss", "rho_lowconf", "rho_patch", "alpha_cov")}
    for n in frames:
        k = names.index(n)
        W, H = size[n]
        d, a = render_depth(gs, torch.from_numpy(c2ws[n]).float().to(device),
                            torch.from_numpy(Ks[n]).float().to(device), W, H, cfg)
        d = F.interpolate(d[None, None], size=(h, w), mode="area")[0, 0].cpu()
        a = F.interpolate(a[None, None], size=(h, w), mode="area")[0, 0].cpu()
        pd = depth[k]
        ok = (a > 0.5) & (d > 1e-6) & (pd > 1e-6)
        inv_r, inv_p = 1.0 / d.clamp(min=1e-6), 1.0 / pd.clamp(min=1e-6)
        m_loss, m_low = ok & hi[k], ok & ~hi[k]
        acc["rho_loss"].append(pearson(inv_r[m_loss], inv_p[m_loss]) if m_loss.sum() >= 16 else np.nan)
        acc["rho_lowconf"].append(pearson(inv_r[m_low], inv_p[m_low]) if m_low.sum() >= 16 else np.nan)
        acc["rho_patch"].append(patch_pearson(inv_r, inv_p, m_loss))
        acc["alpha_cov"].append(float((a > 0.5).float().mean()))
    return {k: float(np.nanmean(v)) for k, v in acc.items()} | {"frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/runs")
    ap.add_argument("--pairs", default=None, help="空格分隔的 <scene>/<run_dir>；不给则按 --lambda 自动配对")
    ap.add_argument("--lambda", dest="lam", default="0.2")
    ap.add_argument("--max_frames", type=int, default=24)
    ap.add_argument("--out", default="/root/autodl-tmp/results/depth_diag.csv")
    args = ap.parse_args()
    device = "cuda"

    if args.pairs:
        runs = args.pairs.split()
    else:
        runs = [f"{s}/{m}_{n}" for s in SCENES for n in NS for m in ("vggt", f"vggt_depth_w{args.lam}")]
    rows = []
    for r in runs:
        d = os.path.join(args.runs, r)
        if not os.path.isdir(os.path.join(d, "gs", "ckpts")):
            print(f"跳过 {r}（没有 ckpt）")
            continue
        res = diag_run(d, args.max_frames, device)
        row = {"run": r} | {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in res.items()}
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
