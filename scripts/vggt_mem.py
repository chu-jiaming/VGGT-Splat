"""测 VGGT 在不同帧数下的显存峰值与推理耗时（D2）。

用法：
    python scripts/vggt_mem.py --image_dir /root/autodl-tmp/data/360_v2/garden/images_4 --frames 16 32 64 100

在 GPU 空闲时跑（先 nvidia-smi 确认没有别的进程）。帧从目录里按文件名排序后均匀抽取。
"""
import argparse
import glob
import os
import time

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 100])
    args = p.parse_args()

    paths = sorted(
        f for f in glob.glob(os.path.join(args.image_dir, "*"))
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not os.path.basename(f).startswith("._")
    )
    print(f"{len(paths)} images in {args.image_dir}")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    torch.cuda.synchronize()
    model_mem = torch.cuda.memory_allocated() / 2**30
    print(f"model weights: {model_mem:.2f} GiB")

    rows = []
    for n in args.frames:
        if n > len(paths):
            print(f"skip N={n}: only {len(paths)} images")
            continue
        idx = torch.linspace(0, len(paths) - 1, n).round().long().tolist()
        images = load_and_preprocess_images([paths[i] for i in idx]).to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                pred = model(images)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            rows.append((n, tuple(images.shape[-2:]), f"{peak:.2f}", f"{time.time() - t0:.1f}"))
            del pred
        except torch.OutOfMemoryError:
            rows.append((n, tuple(images.shape[-2:]), "OOM", "-"))
        del images
        print(rows[-1])

    print("\n| 帧数 | 输入分辨率 (H, W) | 显存峰值 (GiB，含权重) | 推理耗时 (s) |")
    print("| --- | --- | --- | --- |")
    for n, hw, peak, t in rows:
        print(f"| {n} | {hw} | {peak} | {t} |")


if __name__ == "__main__":
    main()
