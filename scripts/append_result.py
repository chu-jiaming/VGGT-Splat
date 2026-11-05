"""把一次运行（场景 × 方法 × N）的结果追加成 results/<scene>.csv 的一行（run_all.sh 调用）。

读取：
    <run_dir>/sparse/0                      训练帧位姿估计（注册数）
    <run_dir>/gs/stats/train_step6999.json  训练集指标、高斯数（只说明拟合）
    <run_dir>/gs/eval.json                  测试集指标（eval.py；已注册测试帧上的平均）
    <run_dir>/timing.tsv                    run_all.sh 记的各阶段耗时（stage<TAB>seconds）

已注册测试帧之外的帧不补值；两个方法在共同注册帧上的比较留到汇总阶段，从 eval.json 的逐帧结果算。
多个任务并行时用文件锁保证一次写一整行。

用法：
    python append_result.py --csv results/lawn.csv --scene lawn --method colmap --n 8 \\
        --run_dir runs/lawn/colmap_8 --train_list data/lawn/splits/train_8.txt --status ok
"""
import argparse
import csv
import fcntl
import json
import os
import time

COLUMNS = [
    "scene", "method", "n", "status", "matcher",
    "train_input", "train_registered", "test_total", "test_registered",
    "psnr_before", "ssim_before", "lpips_before", "psnr_after", "ssim_after", "lpips_after",
    "train_psnr", "num_gs", "align_n_common", "align_center_err_rel", "align_rot_err_deg",
    "pose_s", "joint_s", "train_s", "eval_s", "finished_at", "eval_reason",
]


def count_registered(model_dir):
    if not os.path.isdir(model_dir):
        return 0
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    return sum(1 for im in rec.images.values() if im.has_pose)


def load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--status", required=True, help="ok，或 failed:<阶段>")
    ap.add_argument("--matcher", default="")
    args = ap.parse_args()

    with open(args.train_list) as f:
        train_input = sum(1 for ln in f if ln.strip())
    row = {k: "" for k in COLUMNS}
    row.update(scene=args.scene, method=args.method, n=args.n, status=args.status, matcher=args.matcher,
               train_input=train_input, train_registered=count_registered(os.path.join(args.run_dir, "sparse/0")),
               finished_at=time.strftime("%F %T"))

    stats = load_json(os.path.join(args.run_dir, "gs/stats/train_step6999.json"))
    if stats:
        row.update(train_psnr=f"{stats['psnr']:.3f}", num_gs=stats["num_GS"])

    ev = load_json(os.path.join(args.run_dir, "gs/eval.json"))
    if ev:
        row.update(test_total=ev["n_test"], test_registered=ev["n_registered"])
        if ev.get("status") == "failed":
            row["eval_reason"] = ev.get("reason", "")
            if args.status == "ok":
                row["status"] = "failed:align"
        else:
            for st in ("before", "after"):
                for k in ("psnr", "ssim", "lpips"):
                    v = ev["mean"][st][k]
                    row[f"{k}_{st}"] = "" if v is None else f"{v:.4f}"
            a = ev["align"]
            row.update(align_n_common=a["n_common"], align_center_err_rel=f"{a['center_err_rel_median']:.5f}",
                       align_rot_err_deg=f"{a['rot_err_deg_median']:.4f}")

    timing = os.path.join(args.run_dir, "timing.tsv")
    if os.path.isfile(timing):
        with open(timing) as f:
            for ln in f:
                stage, sec = ln.rstrip("\n").split("\t")
                if f"{stage}_s" in row:
                    row[f"{stage}_s"] = sec

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(" ".join(f"{k}={row[k]}" for k in ("scene", "method", "n", "status", "test_registered", "psnr_after")))


if __name__ == "__main__":
    main()
