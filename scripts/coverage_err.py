"""测试帧误差是否集中在训练帧没看到的区域（D11 失效分析；验证 D10 的结论“瓶颈不在已观测表面的几何”）。

对每个运行的每个已注册测试帧（eval.json 里位姿优化之后的位姿、训练内参中位数）：
  1. 渲染 RGB + 深度，把每个像素反投影成 3D 点
  2. 投到每个训练相机（pose opt 之后）：在画面内、在相机前方、且和该训练视角渲染深度一致（相对差 < --tol，
     即没被遮挡）就算“被这个训练帧看到”
  3. 像素分三类：empty（测试视角渲染 alpha < 0.5，没有几何）、unseen（有几何但没有训练帧看到）、seen（≥ 1 个训练帧）
每类在全部测试帧上汇总 MSE → PSNR，并记像素占比；逐帧再算“unseen 占比”与“该帧 PSNR”的相关。

看到 / 没看到是用 3DGS 自己的几何判断的，几何错的地方会被误判；它是近似，用来看趋势。

输出：--out CSV 每个运行一行；--save_maps 时每帧存 GT | 渲染 | 误差 | 覆盖 四联图到 <run>/gs/coverage_maps/
用法（服务器）：
    ~/work/.venv/bin/python scripts/coverage_err.py --runs /root/autodl-tmp/runs --out /root/autodl-tmp/results/coverage.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import Parser, imageio, load_cfg, load_gaussians, load_joint, load_target, rasterization, to_uint8  # noqa: E402

SCENES = ["desk", "lawn", "room", "garden", "bicycle", "counter"]
NS = ["8", "16", "32", "all"]


def render(gs, c2w, K, W, H, cfg):
    """→ rgb (H, W, 3)、depth (H, W)、alpha (H, W)。"""
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
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], alphas[0, ..., 0]


def backproject(depth, K, c2w):
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    pix = torch.stack([u + 0.5, v + 0.5, torch.ones_like(depth)], -1).reshape(-1, 3)
    cam = (torch.linalg.inv(K) @ pix.T).T * depth.reshape(-1, 1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def seen_count(X, train, tol):
    """X: (P, 3) 世界点；train: [(c2w, K, depth, alpha)] → (P,) 看到它的训练帧数。"""
    cnt = torch.zeros(len(X), dtype=torch.int32, device=X.device)
    for c2w, K, d, a in train:
        w2c = torch.linalg.inv(c2w)
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        uv = (Xc @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]
        H, W = d.shape
        u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        dz = d[vv, uu]
        ok &= (a[vv, uu] > 0.5) & ((z - dz).abs() < tol * z)
        cnt += ok.int()
    return cnt


def psnr(sq_sum, n):
    return float(-10 * np.log10(sq_sum / n)) if n > 0 else float("nan")


@torch.no_grad()
def analyze(run_dir, tol, save_maps, device):
    gs_dir = os.path.join(run_dir, "gs")
    with open(os.path.join(gs_dir, "eval.json")) as f:
        ev = json.load(f)
    if ev.get("status") != "ok":
        return None
    cfg = load_cfg(gs_dir)
    ckpt = torch.load(ev["ckpt"], map_location="cpu", weights_only=False)
    gs = load_gaussians(ckpt, device)
    c2ws, _, Ks = load_target(cfg, ckpt)
    p = Parser(cfg["data_dir"], factor=cfg["data_factor"], normalize=cfg.get("normalize_world_space", True), test_every=0)
    sizes = {n: p.imsize_dict[p.camera_ids[i]] for i, n in enumerate(p.image_names)}
    train = []
    for n, c2w in c2ws.items():
        c = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(Ks[n]).float().to(device)
        _, d, a = render(gs, c, K, *sizes[n], cfg)
        train.append((c, K, d, a))

    joint = load_joint(ev["joint_dir"], ev["data_factor"])
    K_test = torch.tensor(ev["align"]["K_train_median"], dtype=torch.float32, device=device)
    map_dir = os.path.join(gs_dir, "coverage_maps")
    if save_maps:
        os.makedirs(map_dir, exist_ok=True)
    acc = {k: [0.0, 0] for k in ("seen", "unseen", "empty", "all")}
    per_frame = []
    for fr in ev["frames"]:
        if not fr.get("registered"):
            continue
        _, _, (W, H), img_path = joint[fr["name"]]
        gt = torch.from_numpy(imageio.imread(img_path)[..., :3]).float().to(device) / 255.0
        c2w = torch.tensor(fr["c2w_after"], dtype=torch.float32, device=device)
        rgb, d, a = render(gs, c2w, K_test, W, H, cfg)
        cnt = seen_count(backproject(d, K_test, c2w), train, tol).reshape(H, W)
        empty = a < 0.5
        seen = (cnt > 0) & ~empty
        unseen = (cnt == 0) & ~empty
        se = ((rgb - gt) ** 2).mean(-1)
        for k, m in (("seen", seen), ("unseen", unseen), ("empty", empty), ("all", torch.ones_like(empty))):
            acc[k][0] += float(se[m].sum())
            acc[k][1] += int(m.sum())
        per_frame.append((float(unseen.float().mean() + empty.float().mean()), fr["after"]["psnr"]))
        if save_maps:
            err = (se.sqrt() / 0.3).clamp(0, 1)  # RMSE 0.3 以上饱和
            err_rgb = torch.stack([err, err * 0.3, 1 - err], -1)
            cov = torch.stack([unseen.float(), seen.float(), empty.float()], -1)  # 红 unseen 绿 seen 蓝 empty
            canvas = np.concatenate([to_uint8(x[None]) for x in (gt, rgb, err_rgb, cov)], axis=1)
            imageio.imwrite(os.path.join(map_dir, os.path.splitext(fr["name"])[0] + ".png"), canvas)

    total = acc["all"][1]
    frac = np.array([x for x, _ in per_frame])
    ps = np.array([y for _, y in per_frame])
    corr = float(np.corrcoef(frac, ps)[0, 1]) if len(per_frame) > 2 and frac.std() > 0 else float("nan")
    return {
        "frames": len(per_frame),
        **{f"frac_{k}": f"{acc[k][1] / total:.3f}" for k in ("seen", "unseen", "empty")},
        **{f"psnr_{k}": f"{psnr(*acc[k]):.2f}" for k in ("seen", "unseen", "empty", "all")},
        "corr_frac_uncovered_vs_psnr": f"{corr:.3f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/runs")
    ap.add_argument("--list", default=None, help="空格分隔的 <scene>/<run_dir>；默认全部 colmap / vggt / vggt_depth_w0.2")
    ap.add_argument("--tol", type=float, default=0.05, help="遮挡判断的相对深度容差")
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--out", default="/root/autodl-tmp/results/coverage.csv")
    args = ap.parse_args()

    runs = args.list.split() if args.list else [
        f"{s}/{m}_{n}" for s in SCENES for n in NS for m in ("colmap", "vggt", "vggt_depth_w0.2")]
    rows = []
    for r in runs:
        d = os.path.join(args.runs, r)
        if not os.path.isfile(os.path.join(d, "gs", "eval.json")):
            print(f"跳过 {r}（没有 eval.json）")
            continue
        res = analyze(d, args.tol, args.save_maps, "cuda")
        if res is None:
            print(f"跳过 {r}（评测失败）")
            continue
        rows.append({"run": r, **res})
        print(" ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
