#!/usr/bin/env bash
# 换随机种子重复训练，估计同一配置的结果波动（判断零点几 dB 的差是否可信）。
#
# 用法：
#   seed_repeat.sh job <scene> <method> <N> <seed> [λ]   method = vggt | vggt_depth（λ 必填）
#   seed_repeat.sh summary                                 汇总 results/seeds.csv（含原始 seed 42 运行）
#
# 每次运行 runs/<scene>/<run>_s<seed>/：sparse / images / vggt_pred.npz 软链接到 vggt_<N>（位姿、初始点云、
# 深度先验都和原运行相同，只差训练种子），复用 vggt_<N>_joint 做测试帧位姿，评测与 run_all.sh 相同。
# 评测完删掉 ckpt（只留 eval.json、stats、cfg），省数据盘。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/data}
RUNS=${RUNS:-/root/autodl-tmp/runs}
RESULTS=${RESULTS:-/root/autodl-tmp/results}
STOP_STEP=${STOP_STEP:-7000}
CSV=$RESULTS/seeds.csv
export PATH=/root/work/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_HOME=/root/work/torch

factor_of() { if [[ -L $DATA/$1/frames ]]; then echo 1; else echo 2; fi; }

run_name() {  # run_name <method> <N> [λ]
    if [[ $1 == vggt_depth ]]; then echo "vggt_depth_w${3}_$2"; else echo "vggt_$2"; fi
}

record() {  # record <scene> <method> <N> <λ> <seed> <eval.json>
    python - "$@" "$CSV" <<'EOF'
import csv, fcntl, json, os, sys
scene, method, n, lam, seed, ev_path, out = sys.argv[1:]
ev = json.load(open(ev_path))
m = ev["mean"]["after"] if ev.get("status") == "ok" else {"psnr": None, "ssim": None, "lpips": None}
row = {"scene": scene, "method": method, "depth_lambda": lam, "n": n, "seed": seed,
       "n_registered": ev.get("n_registered", 0), **{k: m[k] for k in ("psnr", "ssim", "lpips")}}
with open(out, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    w = csv.DictWriter(f, fieldnames=list(row))
    f.seek(0, os.SEEK_END)
    if f.tell() == 0:  # 在锁内判断：并行任务同时创建文件时只写一次表头
        w.writeheader()
    w.writerow(row)
print(row)
EOF
}

job() {
    local scene=$1 method=$2 n=$3 seed=$4 lambda=${5:-}
    local base src d f ckpt x
    base=$(run_name "$method" "$n" "$lambda")
    src=$RUNS/$scene/vggt_$n
    d=$RUNS/$scene/${base}_s$seed
    f=$(factor_of "$scene")
    local -a extra=(--seed "$seed")
    [[ $method == vggt_depth ]] && extra+=(--dense_depth --dense_depth_lambda "$lambda")
    echo "[$(date '+%F %T')] [START] $scene $base seed=$seed"
    # 原运行（seed 42）先记一行，便于汇总
    if ! grep -q "^$scene,$method,$lambda,$n,42," "$CSV" 2>/dev/null; then
        record "$scene" "$method" "$n" "$lambda" 42 "$RUNS/$scene/$base/gs/eval.json"
    fi
    mkdir -p "$d"
    for x in sparse images vggt_pred.npz; do ln -sfn "$src/$x" "$d/$x"; done
    [[ $f -gt 1 ]] && ln -sfn images "$d/images_$f"
    ckpt=$d/gs/ckpts/ckpt_$((STOP_STEP - 1))_rank0.pt
    if [[ ! -f $d/gs/eval.json ]]; then
        (cd "$REPO/trainer" && python simple_trainer.py default --disable_viewer --test_every 0 --pose_opt \
            --data_factor "$f" --stop_step "$STOP_STEP" --data_dir "$d" --result_dir "$d/gs" "${extra[@]}") \
            > "$d/train.log" 2>&1 || { echo "训练失败，见 $d/train.log"; return 1; }
        python "$REPO/scripts/eval.py" --result_dir "$d/gs" --ckpt "$ckpt" --joint_dir "$RUNS/$scene/vggt_${n}_joint" \
            --test_list "$DATA/$scene/splits/test.txt" --out "$d/gs/eval.json" > "$d/eval.log" 2>&1 \
            || { echo "评测失败，见 $d/eval.log"; return 1; }
    fi
    rm -rf "$d/gs/ckpts" "$d/gs/renders" "$d/images_${f}_png"
    record "$scene" "$method" "$n" "$lambda" "$seed" "$d/gs/eval.json"
    echo "[$(date '+%F %T')] [END] $scene $base seed=$seed"
}

summary() {
    python - "$CSV" <<'EOF'
import csv, sys
from collections import defaultdict
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
g = defaultdict(dict)
for r in rows:
    g[(r["scene"], r["n"], r["method"], r["depth_lambda"])][r["seed"]] = r
print("| 场景 | N | 方法 | 种子数 | PSNR 均值 ± 标准差 | 极差 | SSIM 均值 | LPIPS 均值 | 各种子 PSNR |")
print("|---|---|---|---|---|---|---|---|---|")
for (s, n, m, lam), by_seed in sorted(g.items()):
    p = np.array([float(r["psnr"]) for r in by_seed.values() if r["psnr"]])
    ss = np.mean([float(r["ssim"]) for r in by_seed.values() if r["ssim"]])
    lp = np.mean([float(r["lpips"]) for r in by_seed.values() if r["lpips"]])
    lab = f"vggt_depth λ={lam}" if m == "vggt_depth" else m
    seeds = " ".join(f"{k}:{float(v['psnr']):.2f}" for k, v in sorted(by_seed.items(), key=lambda kv: int(kv[0])) if v["psnr"])
    print(f"| {s} | {n} | {lab} | {len(p)} | {p.mean():.2f} ± {p.std(ddof=1) if len(p) > 1 else 0:.2f} | "
          f"{np.ptp(p):.2f} | {ss:.3f} | {lp:.3f} | {seeds} |")
EOF
}

case ${1:-} in
    job) shift; job "$@" ;;
    summary) summary ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
