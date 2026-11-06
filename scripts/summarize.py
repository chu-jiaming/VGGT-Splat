"""汇总 run_all.sh 的结果：方法对比表 + 帧数–质量曲线 + 失败 / 可疑对齐清单。

读取：
    <results>/<scene>.csv                     append_result.py 的逐次记录（同一组合多行时取最后一行）
    <runs>/<scene>/<run>/gs/eval.json         eval.py 的逐帧指标（算共同注册帧上的平均）

比较规则（11-05 定）：
    - 指标都是测试时位姿优化之后的（after）
    - 未注册的测试帧不补值；方法之间在“共同注册帧”上比：同一场景、同一 N 下，
      所有 status=ok 且对齐可信的运行都注册了的测试帧
    - 对齐可信：align_center_err_rel（中位）≤ --center_thres 且 align_rot_err_deg（中位）≤ --rot_thres；
      不可信的运行单独列出，不进入共同帧比较和曲线。中心误差默认 10%：成功运行最大 7.4%（room vggt_8，
      8 帧 VGGT 的正常精度），两次 COLMAP 重建对不上的是 13% / 76%，10% 落在两者之间的空档

输出（--out，默认 <results>/summary/）：
    summary.md    每个场景一张表（自己注册帧上的平均 + 共同注册帧上的平均）、失败与可疑清单、深度权重扫描
    summary.csv   同样内容的长表，便于再加工
    curves.png    3 个指标 × 场景，x 为训练帧数 N，每条线一个方法，点是共同注册帧上的平均

用法（服务器）：
    ~/work/.venv/bin/python scripts/summarize.py --results /root/autodl-tmp/results --runs /root/autodl-tmp/runs
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

SCENES = ["desk", "lawn", "room", "garden", "bicycle", "counter"]
NS = ["8", "16", "32", "all"]
METRICS = [("psnr", "PSNR ↑", "{:.2f}"), ("ssim", "SSIM ↑", "{:.3f}"), ("lpips", "LPIPS ↓", "{:.3f}")]
# 分类色（dataviz 参考调色板前 3 个槽位，已过 validate_palette.js）+ 不同 marker 作为第二编码
STYLE = {"colmap": ("#2a78d6", "o"), "vggt": ("#eb6834", "s"), "vggt_depth": ("#1baf7a", "^")}


def label_of(row):
    return f"vggt_depth λ={row['depth_lambda']}" if row["method"] == "vggt_depth" else row["method"]


def run_dir_name(row):
    if row["method"] == "vggt_depth":
        return f"vggt_depth_w{row['depth_lambda']}_{row['n']}"
    return f"{row['method']}_{row['n']}"


def load_rows(results_dir):
    """每个 (scene, method, depth_lambda, n) 取 CSV 里最后一行（run_all.sh 续跑会追加新行）。"""
    last = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        with open(path) as f:
            rd = csv.DictReader(f)
            # 只读 append_result.py 写的逐场景表（results/ 下还有 depth_diag / coverage / seeds 等 CSV）
            if not {"scene", "status", "finished_at"} <= set(rd.fieldnames or []):
                continue
            for r in rd:
                last[(r["scene"], r["method"], r.get("depth_lambda", ""), r["n"])] = r
    return list(last.values())


def fnum(s):
    return float(s) if s not in ("", None) else None


def assess(row, args):
    """返回 (可进入比较?, 原因)。"""
    if row["status"] != "ok":
        return False, row["status"] + (f"（{row['eval_reason']}）" if row.get("eval_reason") else "")
    c, r = fnum(row["align_center_err_rel"]), fnum(row["align_rot_err_deg"])
    why = []
    if c is not None and c > args.center_thres:
        why.append(f"中心误差 {c:.1%}")
    if r is not None and r > args.rot_thres:
        why.append(f"旋转误差 {r:.1f}°")
    if why:
        return False, "对齐可疑：" + "，".join(why)
    return True, ""


def per_frame(eval_path):
    """name → after 指标；只含已注册帧。"""
    if not os.path.isfile(eval_path):
        return None
    with open(eval_path) as f:
        ev = json.load(f)
    return {fr["name"]: fr["after"] for fr in ev.get("frames", []) if fr.get("registered")}


def mean_over(frames, names):
    if not frames or not names:
        return {k: None for k, _, _ in METRICS}
    return {k: float(np.mean([frames[n][k] for n in names])) for k, _, _ in METRICS}


def fmt(v, spec):
    return "—" if v is None else spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/root/autodl-tmp/results")
    ap.add_argument("--runs", default="/root/autodl-tmp/runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth_lambdas", default="0.2", help="进入主表和曲线的 vggt_depth 权重（逗号分隔）；其余只进权重扫描表")
    ap.add_argument("--center_thres", type=float, default=0.10, help="对齐中心误差中位数上限（相对轨迹尺度）")
    ap.add_argument("--rot_thres", type=float, default=5.0, help="对齐旋转误差中位数上限（度）")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "summary")
    os.makedirs(out, exist_ok=True)
    main_lambdas = set(args.depth_lambdas.split(","))

    rows = load_rows(args.results)
    for r in rows:
        r["label"] = label_of(r)
        r["ok"], r["why"] = assess(r, args)
        r["frames"] = per_frame(os.path.join(args.runs, r["scene"], run_dir_name(r), "gs/eval.json")) if r["ok"] else None
        if r["ok"] and r["frames"] is None:
            r["ok"], r["why"] = False, "缺 eval.json"
        r["main"] = r["method"] != "vggt_depth" or r["depth_lambda"] in main_lambdas

    scenes = [s for s in SCENES if any(r["scene"] == s for r in rows)] + \
        sorted({r["scene"] for r in rows} - set(SCENES))
    labels = sorted({r["label"] for r in rows if r["main"]},
                    key=lambda x: (list(STYLE).index(x.split()[0]) if x.split()[0] in STYLE else 9, x))

    long_rows, md = [], ["# 结果汇总", "",
                         f"指标为测试时位姿优化之后；“共同帧”= 同场景同 N 下所有可信运行都注册了的测试帧。"
                         f"对齐可信阈值：中心误差 ≤ {args.center_thres:.0%}，旋转误差 ≤ {args.rot_thres}°。", ""]
    curves = {}  # (scene, label) → {n: {metric: v}}
    common_info = {}  # (scene, n) → (n_common, n_test)
    for scene in scenes:
        md += [f"## {scene}", "",
               "| N | 方法 | 状态 | 注册/测试帧 | PSNR | SSIM | LPIPS | 共同帧 | PSNR | SSIM | LPIPS |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in NS:
            group = [r for r in rows if r["scene"] == scene and r["n"] == n and r["main"]]
            if not group:
                continue
            group.sort(key=lambda r: labels.index(r["label"]))
            good = [r for r in group if r["ok"]]
            common = sorted(set.intersection(*(set(r["frames"]) for r in good))) if good else []
            n_test = next((r["test_total"] for r in group if r["test_total"]), "")
            common_info[(scene, n)] = (len(common), n_test)
            for r in group:
                own = mean_over(r["frames"], list(r["frames"] or []))
                com = mean_over(r["frames"], common) if r["ok"] else mean_over(None, [])
                if r["ok"] and common:
                    curves.setdefault((scene, r["label"]), {})[n] = com
                status = "ok" if r["ok"] else r["why"]
                md.append(f"| {n} | {r['label']} | {status} | {r['test_registered'] or 0}/{r['test_total'] or n_test} | "
                          + " | ".join(fmt(own[k], s) for k, _, s in METRICS)
                          + f" | {len(common) if r['ok'] else '—'} | "
                          + " | ".join(fmt(com[k], s) for k, _, s in METRICS) + " |")
                long_rows.append({
                    "scene": scene, "n": n, "method": r["method"], "depth_lambda": r["depth_lambda"],
                    "usable": int(r["ok"]), "note": r["why"],
                    "test_registered": r["test_registered"], "test_total": r["test_total"] or n_test,
                    **{f"{k}_own": own[k] for k, _, _ in METRICS},
                    "n_common": len(common) if r["ok"] else "",
                    **{f"{k}_common": com[k] for k, _, _ in METRICS},
                    "align_center_err_rel": r["align_center_err_rel"], "align_rot_err_deg": r["align_rot_err_deg"],
                })
        md.append("")

    # 失败 / 可疑清单（含不进主表的权重）
    bad = [r for r in rows if not r["ok"]]
    md += ["## 失败与可疑运行", ""]
    if bad:
        md += ["| 场景 | N | 方法 | 原因 | 训练帧注册 | 中心误差 | 旋转误差 |", "|---|---|---|---|---|---|---|"]
        for r in sorted(bad, key=lambda r: (r["scene"], NS.index(r["n"]) if r["n"] in NS else 9, r["label"])):
            md.append(f"| {r['scene']} | {r['n']} | {r['label']} | {r['why']} | {r['train_registered']}/{r['train_input']} | "
                      f"{fmt(fnum(r['align_center_err_rel']), '{:.1%}')} | {fmt(fnum(r['align_rot_err_deg']), '{:.2f}°')} |")
    else:
        md.append("无。")
    md.append("")

    # 深度权重扫描：同场景同 N 下 vggt 与各 λ 的 vggt_depth，在它们的共同帧上比
    scan = [r for r in rows if r["method"] == "vggt_depth"]
    keys = sorted({(r["scene"], r["n"]) for r in scan}, key=lambda k: (k[0], NS.index(k[1]) if k[1] in NS else 9))
    keys = [k for k in keys if sum(r["scene"] == k[0] and r["n"] == k[1] for r in scan) > 1]
    if keys:
        md += ["## 深度权重扫描", "", "同场景同 N 下 vggt 与各 λ 在它们共同注册帧上的平均。", "",
               "| 场景 | N | 方法 | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|---|"]
        for scene, n in keys:
            grp = [r for r in rows if r["scene"] == scene and r["n"] == n and r["method"] in ("vggt", "vggt_depth") and r["ok"]]
            grp.sort(key=lambda r: (r["method"], fnum(r["depth_lambda"]) or 0))
            common = sorted(set.intersection(*(set(r["frames"]) for r in grp))) if grp else []
            for r in grp:
                m = mean_over(r["frames"], common)
                md.append(f"| {scene} | {n} | {r['label']} | " + " | ".join(fmt(m[k], s) for k, _, s in METRICS) + " |")
        md.append("")

    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    plot(curves, common_info, scenes, labels, os.path.join(out, "curves.png"))
    print(f"写入 {out}/summary.md, summary.csv, curves.png")


def plot(curves, common_info, scenes, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图里只用英文：服务器没有中文字体
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3de"
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.edgecolor": grid})
    fig, axes = plt.subplots(len(METRICS), len(scenes), figsize=(3.2 * len(scenes), 2.6 * len(METRICS)),
                             squeeze=False)
    x = np.arange(len(NS))
    for j, scene in enumerate(scenes):
        for i, (k, name, _) in enumerate(METRICS):
            ax = axes[i][j]
            for lab in labels:
                pts = curves.get((scene, lab), {})
                ys = [pts[n][k] if n in pts else np.nan for n in NS]
                color, marker = STYLE.get(lab.split()[0], ("#6b6a63", "D"))
                ax.plot(x, ys, color=color, marker=marker, lw=2, ms=6, mec="white", mew=1.5, label=lab)
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(scene, color=ink, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel(name)
            # 每列自己的刻度（共同帧数各场景不同），只在最下一行写字
            ax.set_xlim(-0.3, len(NS) - 0.7)
            if i == len(METRICS) - 1:
                ax.set_xticks(x, [f"{n}\n{common_info.get((scene, n), ('', ''))[0]}/"
                                  f"{common_info.get((scene, n), ('', ''))[1]}" for n in NS])
            else:
                ax.set_xticks(x, [""] * len(NS))
    axes[-1][0].set_xlabel("training frames N\ncommon / test frames", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=STYLE.get(lab.split()[0], ("#6b6a63", "D"))[0], marker=STYLE.get(lab.split()[0], ("", "D"))[1],
                lw=2, ms=6, mec="white", mew=1.5) for lab in labels]
    fig.legend(h, labels, loc="upper right", ncol=len(labels), frameon=False)
    fig.suptitle("Test-set quality vs. training frames (common registered frames, after test-time pose opt)", x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
