"""VGGT 前馈推理（不做 BA）→ gsplat 可直接读的 COLMAP 场景目录（方法 B）。

替代 VGGT 官方 demo_colmap.py 的无 BA 分支：官方脚本依赖 pycolmap 3.10 的构造 API，在 pycolmap 4.x 上报错；
这里自己写 COLMAP 文本格式（scripts/colmap_io.py），读取仍由 gsplat 用 pycolmap 4.x 完成。

和官方脚本的差别：
- 预处理用 VGGT 默认的 crop 模式（宽缩放到 518，高按比例取 14 的倍数，竖图中心裁到 518），
  不是官方脚本的“补黑边成正方形”，横屏视频不浪费 token；内参按 x / y 各自的缩放比换算回原图分辨率
- 图片按文件名排序后输入，第一帧（世界坐标系参考帧）是确定的
- 权重用 from_pretrained 从 HF 缓存加载，不需要 torch hub 的 model.pt

输出（--out_dir）：
    images/            指向原图的软链接（gsplat 要求 data_dir/images 存在）
    sparse/0/          cameras.txt images.txt points3D.txt（PINHOLE，原图分辨率）
    points.ply         同一份点云，方便用 MeshLab 等查看
    vggt_pred.npz      原始预测（模型分辨率的深度/置信度 + 原图分辨率的内参），留给 M2 深度监督

用法：
    python scripts/vggt_to_colmap.py --image_dir /root/autodl-tmp/data/vggt_garden16/images --out_dir /root/autodl-tmp/data/vggt_garden16_out
    # 只用部分帧：--list splits/train_16.txt（每行一个文件名）
"""
import argparse
import os
import time

import numpy as np
from PIL import Image as PILImage

from colmap_io import Camera, Image, rotmat_to_qvec, write_model_text

VGGT_WIDTH = 518


def crop_geometry(width, height):
    """复现 vggt.utils.load_fn.load_and_preprocess_images(mode="crop") 的尺寸计算。

    返回 (new_h, start_y, sx, sy)：模型输入为 min(new_h, 518) x 518；
    模型像素坐标 (x, y) 对应原图 ((x) * sx, (y + start_y) * sy)（均以左上角为原点的连续坐标）。
    """
    new_h = round(height * (VGGT_WIDTH / width) / 14) * 14
    start_y = (new_h - VGGT_WIDTH) // 2 if new_h > VGGT_WIDTH else 0
    return new_h, start_y, width / VGGT_WIDTH, height / new_h


def load_images(paths):
    """返回 (images (S, 3, h, 518) float32 in [0,1], (W, H))。要求所有图同尺寸（视频帧）。"""
    sizes = {PILImage.open(p).size for p in paths}
    if len(sizes) != 1:
        raise ValueError(f"图片尺寸不一致：{sizes}")
    W, H = sizes.pop()
    new_h, start_y, _, _ = crop_geometry(W, H)
    out_h = min(new_h, VGGT_WIDTH)
    arr = np.empty((len(paths), 3, out_h, VGGT_WIDTH), dtype=np.float32)
    for i, p in enumerate(paths):
        img = PILImage.open(p).convert("RGB").resize((VGGT_WIDTH, new_h), PILImage.Resampling.BICUBIC)
        arr[i] = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[:, start_y:start_y + out_h]
    return arr, (W, H)


def intrinsics_to_original(intrinsic, width, height):
    """模型分辨率的 K (S, 3, 3) → 原图分辨率的 K。"""
    _, start_y, sx, sy = crop_geometry(width, height)
    K = intrinsic.astype(np.float64).copy()
    K[:, 1, 2] += start_y
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    return K


def build_model(names, size, extrinsic, intrinsic, depth, conf, colors, conf_thres, max_points, shared_camera, rng):
    """由 VGGT 预测构造 COLMAP 模型（cameras, images, xyz, rgb）。

    extrinsic (S, 3, 4) w2c，intrinsic (S, 3, 3) 模型分辨率，depth / conf (S, h, w)，colors (S, h, w, 3) uint8。
    每个 3D 点由一帧的一个像素反投影得到，track 只含这一帧的这个像素。
    """
    S, h, w = depth.shape
    W, H = size
    _, start_y, sx, sy = crop_geometry(W, H)

    # 反投影：像素中心 (u+0.5, v+0.5)，与 COLMAP 的像素约定一致
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=-1)  # (S, h, w, 3)
    R, t = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    world = np.einsum("sij,shwj->shwi", R.transpose(0, 2, 1), cam - t[:, None, None, :])

    mask = (conf >= conf_thres) & (depth > 0)
    idx = np.flatnonzero(mask)
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    f_idx, v_idx, u_idx = np.unravel_index(idx, mask.shape)
    xyz = world.reshape(-1, 3)[idx]
    rgb = colors.reshape(-1, 3)[idx]
    print(f"{mask.sum()} / {mask.size} 像素通过置信度阈值 {conf_thres}，保留 {len(idx)} 个点")

    K = intrinsics_to_original(intrinsic, W, H)
    if shared_camera:
        fx_m, fy_m = K[:, 0, 0].mean(), K[:, 1, 1].mean()
        cameras = [Camera(1, "PINHOLE", W, H, np.array([fx_m, fy_m, K[0, 0, 2], K[0, 1, 2]]))]
    else:
        cameras = [Camera(i + 1, "PINHOLE", W, H, np.array([K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]])) for i in range(S)]

    images = []
    point_ids = np.arange(1, len(idx) + 1)
    for i in range(S):
        sel = f_idx == i
        xys = np.stack([(u_idx[sel] + 0.5) * sx, (v_idx[sel] + 0.5 + start_y) * sy], axis=-1)
        images.append(Image(
            id=i + 1, qvec=rotmat_to_qvec(R[i]), tvec=t[i].astype(np.float64),
            camera_id=1 if shared_camera else i + 1, name=names[i],
            xys=xys, point3D_ids=point_ids[sel],
        ))
    return cameras, images, xyz, rgb, K


def write_ply(path, xyz, rgb):
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def run_vggt(images_np):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = torch.from_numpy(images_np).to(device)[None]  # (1, S, 3, h, w)

    t0 = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            tokens, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, images, ps_idx)
    torch.cuda.synchronize()
    print(f"VGGT 推理 {images.shape[1]} 帧，{time.time() - t0:.1f} s，显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    to_np = lambda x: x.squeeze(0).float().cpu().numpy()
    return to_np(extrinsic), to_np(intrinsic), to_np(depth)[..., 0], to_np(depth_conf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--list", help="只用这些帧：每行一个 image_dir 下的文件名")
    p.add_argument("--conf_thres", type=float, default=5.0, help="深度置信度阈值（同官方脚本默认值）")
    p.add_argument("--max_points", type=int, default=100000, help="点云随机下采样上限（同官方脚本）")
    p.add_argument("--shared_camera", action="store_true", help="所有帧共用一个相机（fx/fy 取平均）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.list:
        with open(args.list) as f:
            names = sorted(line.strip() for line in f if line.strip())
    else:
        names = sorted(
            f for f in os.listdir(args.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._")
        )
    paths = [os.path.join(args.image_dir, n) for n in names]
    images_np, size = load_images(paths)
    print(f"{len(names)} 张图，原图 {size[0]}x{size[1]} → 模型输入 {images_np.shape[-1]}x{images_np.shape[-2]}")

    extrinsic, intrinsic, depth, conf = run_vggt(images_np)
    colors = (images_np.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
    cameras, images, xyz, rgb, K = build_model(
        names, size, extrinsic, intrinsic, depth, conf, colors,
        args.conf_thres, args.max_points, args.shared_camera, np.random.default_rng(args.seed),
    )

    sparse_dir = os.path.join(args.out_dir, "sparse", "0")
    write_model_text(sparse_dir, cameras, images, xyz, rgb)
    write_ply(os.path.join(args.out_dir, "points.ply"), xyz, rgb)
    _, start_y, _, _ = crop_geometry(*size)
    np.savez_compressed(
        os.path.join(args.out_dir, "vggt_pred.npz"),
        names=np.array(names), extrinsic=extrinsic, intrinsic_model=intrinsic, K=K,
        depth=depth.astype(np.float32), conf=conf.astype(np.float32),
        image_size=np.array(size), crop_start_y=start_y,
    )

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for n, src in zip(names, paths):
        dst = os.path.join(img_dir, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)

    print(f"写出 {sparse_dir}（{len(images)} 帧，{len(xyz)} 点）")
    print(f"fx 范围 {K[:, 0, 0].min():.1f} – {K[:, 0, 0].max():.1f}（原图分辨率）")


if __name__ == "__main__":
    main()
