"""把一个或两个 COLMAP 模型的相机（中心 + 朝向）和稀疏点云画成 PNG，调试位姿用（roadmap L1 产出）。

给了 --other 时，按图像文件名配对，用相机中心做 Umeyama Sim(3) 对齐（other → model），两组相机画在同一张图里；
点云只画 --model 的。

两个子图：
- 左：3D 透视
- 右：俯视。投影到相机中心的前两个主成分平面上（环绕拍摄时就是“从上往下看”），轨迹形状和朝向一眼可见

相机画成一个点 + 一条沿光轴（相机 +z，OpenCV 约定）的短线；三角形标第一帧。
朝向全部指向场景中心 → w2c/c2w 没搞反；朝向全部背离 → 很可能把 c2w 当 w2c 写了。

用法：
    python plot_cameras.py --model /root/autodl-tmp/runs/desk/vggt_16/sparse/0 --out desk_vggt16.png
    python plot_cameras.py --model /root/autodl-tmp/runs/_trial/colmap_desk_all/sparse/0 \
        --other /root/autodl-tmp/runs/desk/vggt_16/sparse/0 --labels colmap_all vggt_16 --out desk_cmp.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from compare_poses import load, umeyama


def load_points(path, max_points, rng):
    rec = pycolmap.Reconstruction(path)
    pts = list(rec.points3D.values())
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xyz = np.stack([p.xyz for p in pts])
    rgb = np.stack([p.color for p in pts]) / 255.0
    if len(xyz) > max_points:
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def cameras(poses, names):
    """返回 (C (N, 3), d (N, 3))：相机中心与世界系下的光轴方向（R_w2c 的第 3 行）。"""
    C = np.stack([poses[n][1] for n in names])
    d = np.stack([poses[n][0][2] for n in names])
    return C, d


def draw(ax, proj, C, d, color, label, length):
    """proj: (3, k) 投影矩阵，k = 3 画 3D，k = 2 画俯视。"""
    P, E = C @ proj, (C + length * d) @ proj
    ax.plot(*P.T, "-", color=color, lw=0.8, alpha=0.6)
    ax.scatter(*P.T, color=color, s=10, label=label)
    for p, e in zip(P, E):
        ax.plot(*np.stack([p, e]).T, "-", color=color, lw=1.2)
    ax.scatter(*P[0], color=color, marker="^", s=60, edgecolors="k")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="主模型目录（坐标系以它为准，点云取它的）")
    p.add_argument("--other", help="第二个模型目录，Sim(3) 对齐到 --model 后一起画")
    p.add_argument("--labels", nargs=2, default=["model", "other"])
    p.add_argument("--out", required=True, help="输出 PNG")
    p.add_argument("--max_points", type=int, default=20000, help="点云随机下采样上限，0 = 不画点")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    poses, _ = load(args.model)
    names = sorted(poses)
    C, d = cameras(poses, names)
    groups = [(C, d, "tab:blue", f"{args.labels[0]} ({len(names)})")]

    if args.other:
        other, _ = load(args.other)
        common = sorted(set(poses) & set(other))
        if len(common) < 3:
            raise SystemExit(f"共同帧 {len(common)} < 3，无法对齐")
        s, R, t = umeyama(np.stack([other[n][1] for n in common]), np.stack([poses[n][1] for n in common]))
        C_o, d_o = cameras(other, sorted(other))
        groups.append((s * C_o @ R.T + t, d_o @ R.T, "tab:red", f"{args.labels[1]} ({len(other)}, Sim(3) s={s:.3f})"))
        print(f"共同 {len(common)} 帧，Sim(3) 尺度 {s:.4f}")

    xyz, rgb = load_points(args.model, args.max_points, rng) if args.max_points > 0 else (np.zeros((0, 3)),) * 2

    # 俯视平面：主模型相机中心的前两个主成分
    mu = C.mean(0)
    _, _, Vt = np.linalg.svd(C - mu)
    top = Vt[:2].T
    scale = np.median(np.linalg.norm(C - mu, axis=1))
    length = 0.15 * scale

    # 坐标范围：相机 + 点云中离相机质心最近的 95%，飞点不撑大画面
    if len(xyz):
        dist = np.linalg.norm(xyz - mu, axis=1)
        keep = dist <= np.percentile(dist, 95)
        xyz, rgb = xyz[keep], rgb[keep]
    allpts = np.concatenate([g[0] for g in groups] + [xyz])

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    for ax, proj in [(ax3, np.eye(3)), (ax2, top)]:
        if len(xyz):
            ax.scatter(*(xyz @ proj).T, c=rgb, s=0.3, alpha=0.5, linewidths=0)
        for g in groups:
            draw(ax, proj, *g, length)

    lo, hi = allpts.min(0), allpts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax3.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(c[2] - r, c[2] + r), xlabel="x", ylabel="y", zlabel="z")
    ax3.set_title("3D (triangle = first frame, line = optical axis +z)")
    ax2.set_aspect("equal")
    ax2.set(xlabel="PC1", ylabel="PC2", title="Top view (PCA plane of camera centers)")
    ax2.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"写出 {args.out}（{len(xyz)} 个点）")


if __name__ == "__main__":
    main()
