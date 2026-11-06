"""把训练好的 3DGS ckpt 导出成网页查看器能打开的文件（D11）。

每个运行读 <runs>/<scene>/<run>/gs/ckpts/ckpt_6999_rank0.pt，用 gsplat.export_splats 写：
    <out>/<scene>_<run>.splat           antimatter15 splat 格式（每个高斯 32 字节，只有 0 阶颜色）
    <out>/<scene>_<run>.compressed.ply  SuperSplat 压缩 PLY（保留高阶球谐）
    --full_ply 时再写 <out>/<scene>_<run>.ply（标准 PLY，体积约为 .splat 的 8 倍）

坐标是训练时归一化后的世界坐标（gsplat Parser normalize），和 ckpt 里一致。

用法（服务器）：
    ~/work/.venv/bin/python scripts/export_splats.py --list "garden/vggt_all garden/colmap_all"
"""
import argparse
import os

import torch
from gsplat import export_splats

FORMATS = {"splat": ".splat", "ply_compressed": ".compressed.ply"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/runs")
    ap.add_argument("--out", default="/root/autodl-tmp/results/export")
    ap.add_argument("--list", required=True, help="空格分隔的 <scene>/<run_dir>")
    ap.add_argument("--step", type=int, default=6999)
    ap.add_argument("--full_ply", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    formats = dict(FORMATS, **({"ply": ".ply"} if args.full_ply else {}))

    for run in args.list.split():
        ckpt = os.path.join(args.runs, run, f"gs/ckpts/ckpt_{args.step}_rank0.pt")
        sp = torch.load(ckpt, map_location="cpu", weights_only=False)["splats"]
        name = run.replace("/", "_")
        sizes = []
        for fmt, ext in formats.items():
            path = os.path.join(args.out, name + ext)
            export_splats(means=sp["means"], scales=sp["scales"], quats=sp["quats"], opacities=sp["opacities"],
                          sh0=sp["sh0"], shN=sp["shN"], format=fmt, save_to=path)
            sizes.append(f"{ext} {os.path.getsize(path) / 2**20:.0f} MB")
        print(f"{run}: {len(sp['means'])} 个高斯；" + "，".join(sizes), flush=True)


if __name__ == "__main__":
    main()
