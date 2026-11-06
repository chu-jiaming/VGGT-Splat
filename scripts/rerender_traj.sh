#!/usr/bin/env bash
# 从已有 ckpt 重渲染更慢的轨迹视频（项目页用），不重新训练。
#
# 用法：
#   rerender_traj.sh <scene>/<run> [<scene>/<run> ...]
#   TRAJ_FRAMES=600 rerender_traj.sh garden/vggt_all
#
# 训练时 render_traj 默认约 150 帧（30 fps 5 s）；全部帧训练时每两个训练相机之间只插 1 帧，相机逐个跳，看起来太快。
# 这里用 trainer --ckpt（只评测 + 渲染）配 --traj_frames 重渲染，data_factor 取自原运行的 gs/cfg.yml。
# 输出：<run>/gs_<TRAJ_FRAMES>/videos/traj_<step>.mp4（左 RGB 右深度），日志 <run>/gs_<TRAJ_FRAMES>.log；原 gs/ 不动。
# 渲染路径只取决于训练相机位姿，同一场景两个方法的帧数相同时视角同步（COLMAP 没注册全部帧时不同步）。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-/root/autodl-tmp/runs}
TRAJ_FRAMES=${TRAJ_FRAMES:-450}
STEP=${STEP:-6999}
export PATH=/root/work/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_HOME=/root/work/torch

[[ $# -gt 0 ]] || { sed -n 2,5p "$0"; exit 1; }
for run in "$@"; do
    d=$RUNS/$run
    f=$(awk '/^data_factor:/ {print $2}' "$d/gs/cfg.yml")
    out=$d/gs_$TRAJ_FRAMES
    echo "== $run (data_factor $f) → $out"
    if (cd "$REPO/trainer" && python simple_trainer.py default --disable_viewer --test_every 0 --pose_opt \
            --data_factor "$f" --data_dir "$d" --result_dir "$out" \
            --ckpt "$d/gs/ckpts/ckpt_${STEP}_rank0.pt" --traj_frames "$TRAJ_FRAMES") > "$out.log" 2>&1; then
        ls "$out/videos"
    else
        echo "FAILED（见 $out.log）"
        tail -5 "$out.log"
    fi
done
