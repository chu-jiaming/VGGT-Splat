"""两个方法的 3DGS 沿同一条椭圆环绕路径渲染（项目页的新视角对比视频，适合 360° 拍摄的场景）。

训练轨迹视频（render_traj interp）沿各自估计的训练相机走：视角都靠近训练帧、速度随手持拍摄忽快忽慢，
两个方法的相机估计不同处还会各看各的。这里改成：
  1. 读两个运行 pose opt 之后的训练相机（eval.load_target，各自的归一化坐标系）
  2. 在参考运行（--ref，默认 COLMAP）坐标系里生成椭圆路径：gsplat generate_ellipse_path_z，
     焦点 = 训练相机光轴最近点，半径 = 相机水平偏移的 90 百分位 × --radius_scale，高度 = 相机平均高度
  3. 用共同训练帧拟合 参考 → 另一个运行 的 Sim(3)（eval.orient_center_sim3，旋转由朝向、尺度平移由中心），
     把路径映射过去
  4. 两个模型都用参考运行的内参（中位数）渲染，画面范围一致；冻结高斯，只渲染
输出 <out>/<scene>_orbit_<method>.mp4（30 fps，RGB）和 <scene>_orbit.json（Sim(3) 诊断）。

用法（服务器）：
    ~/work/.venv/bin/python scripts/render_orbit.py --scene garden --frames 600
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import (Parser, imageio, latest_ckpt, load_cfg, load_gaussians, load_target,  # noqa: E402
                  orient_center_sim3, render, rot_angle_deg, to_uint8)
from datasets.traj import focus_point_fn, generate_ellipse_path_z, viewmatrix  # noqa: E402


def load_run(run_dir, device):
    res = os.path.join(run_dir, "gs")
    cfg = load_cfg(res)
    ckpt = torch.load(latest_ckpt(res), map_location="cpu", weights_only=False)
    poses, scale, Ks = load_target(cfg, ckpt)
    p = Parser(cfg["data_dir"], factor=cfg["data_factor"], normalize=cfg.get("normalize_world_space", True),
               test_every=0)
    W, H = p.imsize_dict[p.camera_ids[0]]
    return dict(cfg=cfg, gs=load_gaussians(ckpt, device), poses=poses, scale=scale, Ks=Ks, W=W, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/runs")
    ap.add_argument("--out", default="/root/autodl-tmp/results/orbit")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ref", default="colmap_all", help="生成路径、提供内参的运行")
    ap.add_argument("--other", default="vggt_all")
    ap.add_argument("--frames", type=int, default=600, help="一整圈的帧数（30 fps）")
    ap.add_argument("--radius_scale", type=float, default=1.0)
    ap.add_argument("--height_offset", type=float, default=0.0, help="路径高度偏移，单位为参考运行的轨迹尺度")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    ref = load_run(os.path.join(args.runs, args.scene, args.ref), dev)
    oth = load_run(os.path.join(args.runs, args.scene, args.other), dev)

    # 参考 → 另一个运行的 Sim(3)，共同训练帧拟合
    common = sorted(set(ref["poses"]) & set(oth["poses"]))
    P_r = np.stack([ref["poses"][n] for n in common])
    P_o = np.stack([oth["poses"][n] for n in common])
    s, R, t = orient_center_sim3(P_r[:, :3, :3], P_r[:, :3, 3], P_o[:, :3, :3], P_o[:, :3, 3])
    mapped_c = (s * P_r[:, :3, 3] @ R.T) + t
    err_c = np.linalg.norm(mapped_c - P_o[:, :3, 3], axis=1) / oth["scale"]
    err_r = [rot_angle_deg((R @ a).T @ b) for a, b in zip(P_r[:, :3, :3], P_o[:, :3, :3])]
    diag = {"n_common": len(common), "scale": s, "center_err_rel_median": float(np.median(err_c)),
            "rot_err_deg_median": float(np.median(err_r))}
    print(f"Sim(3) {args.ref} → {args.other}：共同帧 {len(common)}，中心误差中位 {diag['center_err_rel_median']:.2%}，"
          f"旋转误差中位 {diag['rot_err_deg_median']:.2f}°", flush=True)

    # 椭圆路径（参考坐标系）
    ref_poses = np.stack(list(ref["poses"].values()))
    height = ref_poses[:, 2, 3].mean() + args.height_offset * ref["scale"]
    path = generate_ellipse_path_z(ref_poses[:, :3, :], n_frames=args.frames, height=height)  # (F, 3, 4)
    if args.radius_scale != 1.0:
        # 水平方向绕焦点缩放，再重新对准焦点（up 沿用路径自己的 y 轴方向，和 generate_ellipse_path_z 一致）
        focus = focus_point_fn(ref_poses[:, :3, :])
        pos = path[:, :3, 3].copy()
        pos[:, :2] = focus[:2] + (pos[:, :2] - focus[:2]) * args.radius_scale
        y = path[0, :3, 1]
        i = int(np.argmax(np.abs(y)))
        up = np.eye(3)[i] * np.sign(y[i])
        path = np.stack([viewmatrix(focus - p, up, p) for p in pos])
    bottom = np.broadcast_to(np.array([0, 0, 0, 1.0]), (len(path), 1, 4))
    path_ref = np.concatenate([path, bottom], 1)
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = s * R, t
    path_oth = S @ path_ref
    path_oth[:, :3, :3] /= s

    K_med = np.median(np.stack(list(ref["Ks"].values())), 0)
    K = torch.tensor(K_med, dtype=torch.float32, device=dev)
    W, H = ref["W"], ref["H"]
    for name, run, P in [(args.ref, ref, path_ref), (args.other, oth, path_oth)]:
        path_out = os.path.join(args.out, f"{args.scene}_orbit_{name}.mp4")
        # macro_block_size=1：不把奇数尺寸拉伸到 16 的倍数（会轻微变形）；网页用的视频另外重编码
        writer = imageio.get_writer(path_out, fps=30, macro_block_size=1, quality=9)
        with torch.no_grad():
            for c2w in P:
                img = render(run["gs"], torch.tensor(c2w, dtype=torch.float32, device=dev)[None], K, W, H, run["cfg"])
                # H.264 要求偶数宽高：奇数时裁掉最后一列 / 一行（不拉伸）
                writer.append_data(to_uint8(img)[: H - H % 2, : W - W % 2])
        writer.close()
        print(f"写入 {path_out}（{len(P)} 帧，{W}×{H}）", flush=True)
    with open(os.path.join(args.out, f"{args.scene}_orbit.json"), "w") as f:
        json.dump(dict(vars(args), **diag), f, indent=2)


if __name__ == "__main__":
    main()
