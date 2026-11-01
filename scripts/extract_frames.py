"""视频抽帧 → 标记模糊帧 → 生成测试集与各 N 的训练集（M1 ①）。

目录约定见 docs/roadmap.md 的 M0：
    data/<scene>/video.<ext>          原始视频
    data/<scene>/frames/NNNN.jpg      ffmpeg fps=3 全部抽出的帧，从 0001 编号，不删帧
    data/<scene>/splits/              本脚本生成：
        blur.tsv                      每帧模糊度、邻域中位数、比值、是否模糊
        blurry.txt                    被跳过的模糊帧
        test.txt                      去模糊后每 test_every 帧取 1 帧（与 gsplat test_every=8 一致）
        train_{8,16,32}.txt           在训练池（去模糊后除测试帧外）里均匀抽 N 帧
        train_all.txt                 整个训练池；超过 max_all（VGGT 上限 180）时均匀抽 max_all 帧
    每个 split 文件每行一个 frames/ 下的文件名，按文件名排序。

模糊判定：模糊度 > blur_ratio × 前后各 window 帧（含自身）的模糊度中位数。
模糊度用“再模糊”法（Crété-Roffet et al. 2007, The Blur Effect）：把图再做一次方框模糊，
看相邻像素差还剩多少；清晰图损失大、本来就糊的图损失小。取值 0–1，越大越糊。
不用拉普拉斯方差：它主要反映纹理多少，room 的白墙段比书架段低两个数量级，
即使和邻居比，镜头从书架转向白墙时也会被误判（11-01 在 room 上逐帧看过）；再模糊法对纹理多少基本不敏感。
仍和邻居比而不是统一阈值：模糊度的基线随场景变化（lawn 约 0.19，desk 约 0.37，room 约 0.43）。
连续一整段都糊的帧去不掉，这是已知限制。

frames/ 已存在且非空时不重新抽帧（帧编号是所有 split 和运行的基准，不应变化）；split 每次重算覆盖。

用法：
    python scripts/extract_frames.py data/desk
    python scripts/extract_frames.py data/desk data/lawn data/room --blur_ratio 0.5
"""
import argparse
import glob
import os
import subprocess

import numpy as np
from PIL import Image

BLUR_WIDTH = 960  # 统一缩放到这个宽度再算模糊度，结果与原分辨率无关、也更快
REBLUR_SIZE = 9   # 再模糊的方框核长度（原文取 9）


def extract(scene_dir, fps):
    frames_dir = os.path.join(scene_dir, "frames")
    if os.path.isdir(frames_dir) and os.listdir(frames_dir):
        return
    videos = glob.glob(os.path.join(scene_dir, "video.*"))
    if len(videos) != 1:
        raise SystemExit(f"{scene_dir}：没有 frames/，且 video.* 不是恰好一个：{videos}")
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", videos[0],
         "-vf", f"fps={fps}", "-q:v", "2", os.path.join(frames_dir, "%04d.jpg")],
        check=True,
    )


def box_blur(a, axis, k=REBLUR_SIZE):
    """沿 axis 做长度 k 的一维方框模糊，边缘复制填充，输出尺寸不变。"""
    pad = [(k // 2 + 1, k // 2) if i == axis else (0, 0) for i in range(a.ndim)]
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    n = c.shape[axis]
    return (np.take(c, range(k, n), axis=axis) - np.take(c, range(0, n - k), axis=axis)) / k


def blur_score(path):
    """Crété-Roffet 模糊度：横、竖两个方向分别算，取较大值；0–1，越大越糊。"""
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((BLUR_WIDTH, round(h * BLUR_WIDTH / w)), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float64)
    scores = []
    for axis in (0, 1):
        d = np.abs(np.diff(a, axis=axis))
        d_reblur = np.abs(np.diff(box_blur(a, axis), axis=axis))
        lost = np.maximum(0, d - d_reblur).sum()
        scores.append(1 - lost / max(d.sum(), 1e-9))
    return max(scores)


def local_median(x, window):
    return np.array([np.median(x[max(0, i - window): i + window + 1]) for i in range(len(x))])


def uniform(items, n):
    """均匀抽 n 个（含首尾）；不足 n 个时全部返回。"""
    if len(items) <= n:
        return list(items)
    idx = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    return [items[i] for i in idx]


def write_list(path, names):
    with open(path, "w") as f:
        f.writelines(name + "\n" for name in names)


def process(scene_dir, args):
    extract(scene_dir, args.fps)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(scene_dir, "frames", "*.jpg")))
    score = np.array([blur_score(os.path.join(scene_dir, "frames", n)) for n in names])
    ref = local_median(score, args.window)
    ratio = score / ref
    blurry = ratio > args.blur_ratio

    sharp = [n for n, b in zip(names, blurry) if not b]
    test = sharp[::args.test_every]
    pool = [n for i, n in enumerate(sharp) if i % args.test_every != 0]
    splits = {"test": test, "blurry": [n for n, b in zip(names, blurry) if b]}
    for n in args.ns:
        splits[f"train_{n}"] = uniform(pool, n)
    splits["train_all"] = uniform(pool, args.max_all)

    assert not set(test) & set(pool), "测试帧与训练帧有交集"
    for key, split in splits.items():
        assert split == sorted(split), key

    out = os.path.join(scene_dir, "splits")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "blur.tsv"), "w") as f:
        f.write("frame\tblur\tlocal_median\tratio\tblurry\n")
        for n, s, r, q, b in zip(names, score, ref, ratio, blurry):
            f.write(f"{n}\t{s:.4f}\t{r:.4f}\t{q:.3f}\t{int(b)}\n")
    for key, split in splits.items():
        write_list(os.path.join(out, f"{key}.txt"), split)

    scene = os.path.basename(os.path.normpath(scene_dir))
    sizes = "  ".join(f"{k}={len(v)}" for k, v in splits.items() if k != "blurry")
    print(f"{scene}: frames={len(names)}  blurry={blurry.sum()}  pool={len(pool)}  {sizes}")
    print(f"  blurry: {' '.join(n[:-4] for n in splits['blurry']) or '-'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene_dirs", nargs="+", help="data/<scene>，可多个")
    parser.add_argument("--fps", type=float, default=3, help="仅在 frames/ 不存在时使用")
    parser.add_argument("--blur_ratio", type=float, default=1.15,
                        help="模糊度超过邻域中位数的这个倍数算模糊；1.15 在 desk/room 上逐帧看过")
    parser.add_argument("--window", type=int, default=7, help="邻域半径（帧）；3 fps 下 7 帧约 ±2.3 s")
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--ns", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max_all", type=int, default=180, help="VGGT 帧数上限（10-29 实测）")
    args = parser.parse_args()
    for scene_dir in args.scene_dirs:
        process(scene_dir, args)


if __name__ == "__main__":
    main()
