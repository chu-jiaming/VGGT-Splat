#!/usr/bin/env bash
# 批处理（M1 ⑤）：场景 × 方法 × N，每次运行 = 训练帧位姿估计 → 联合运行 → 3DGS 训练 → 测试集评测 → 追加 CSV。
#
# 用法：
#   run_all.sh                                  按下面的环境变量跑全部组合（N 小的先跑）
#   SCENES=lawn NS=8 run_all.sh                 只跑 lawn × N=8 的两个方法
#   run_all.sh job <scene> <method> <N>         只跑一次（xargs 并行时内部调用）
#
# 环境变量（括号内为默认）：
#   SCENES (desk lawn room garden)  METHODS (colmap vggt)  NS (8 16 32 all)
#   JOBS (2)       同时进行的运行数；位姿估计（VGGT / COLMAP，占 GPU、时间短）用文件锁串行，训练并行
#   STOP_STEP (7000)  训练按 30k 调度跑到这一步停（trainer --stop_step，11-05 定）
#   VGGT_ARGS (--shared_camera --conf_percentile 50)  训练帧和联合运行都用
#   DATA / RUNS / RESULTS (/root/autodl-tmp/{data,runs,results})
#
# 每次运行（runs/<scene>/<method>_<N>/）：
#   sparse/0, images/      训练帧位姿（colmap: run_colmap.sh exhaustive；vggt: vggt_to_colmap.py）
#   gs/                    3DGS（--test_every 0 --pose_opt，ckpt_<STOP_STEP-1>）
#   gs/eval.json           eval.py（默认预算 500 步 lr 1e-3，测试帧内参用训练运行中位数）
#   timing.tsv, train.log, eval.log
#   ../<method>_<N>_joint/ 训练帧 + 测试帧联合运行（只拿测试帧初始位姿）
#   ../<method>_<N>.log    这次运行的全部输出
# 结果：results/<scene>.csv 每次运行一行（append_result.py）；失败也记一行，status=failed:<阶段>
#
# 可续跑：已有的位姿估计目录、联合运行目录、ckpt 直接复用；评测和 CSV 每次重做（CSV 会多一行，汇总时取最后一行）。
# 分辨率：自拍场景 data_factor 2（960×540）；garden 的 frames/ 指向 images_4（1237×822），data_factor 1。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/data}
RUNS=${RUNS:-/root/autodl-tmp/runs}
RESULTS=${RESULTS:-/root/autodl-tmp/results}
STOP_STEP=${STOP_STEP:-7000}
POSE_LOCK=${POSE_LOCK:-/tmp/vggt_splat_pose.lock}
# 与 desk vggt_16（11-02）一致：一台相机；默认 conf_thres 5 在 desk / lawn 上一个点都不剩
VGGT_ARGS=${VGGT_ARGS:---shared_camera --conf_percentile 50}
export PATH=/root/work/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_HOME=/root/work/torch
export HF_HOME=/root/work/hf HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com

factor_of() { case $1 in garden) echo 1 ;; *) echo 2 ;; esac; }

has_model() { [[ -f $1/sparse/0/images.txt || -f $1/sparse/0/images.bin ]]; }

timed() {  # timed <run_dir> <stage> <命令...>：运行并把耗时追加到 <run_dir>/timing.tsv
    local d=$1 stage=$2; shift 2
    local t0=$SECONDS rc
    "$@"; rc=$?
    mkdir -p "$d"
    printf '%s\t%s\n' "$stage" $((SECONDS - t0)) >> "$d/timing.tsv"
    return $rc
}

estimate() {  # estimate <method> <out_dir> <list>
    local method=$1 out=$2 list=$3 frames=$DATA/$scene/frames
    if [[ $method == colmap ]]; then
        MATCHER=exhaustive "$REPO/scripts/run_colmap.sh" "$frames" "$out" "$list"
    else
        # shellcheck disable=SC2086
        python "$REPO/scripts/vggt_to_colmap.py" --image_dir "$frames" --out_dir "$out" --list "$list" $VGGT_ARGS
    fi
}

train_gs() {  # train_gs <run_dir> <factor>
    local d=$1 f=$2
    [[ $f -gt 1 ]] && ln -sfn images "$d/images_$f"
    (cd "$REPO/trainer" && python simple_trainer.py default --disable_viewer --test_every 0 --pose_opt \
        --data_factor "$f" --stop_step "$STOP_STEP" --data_dir "$d" --result_dir "$d/gs") > "$d/train.log" 2>&1
}

job() {
    scene=$1; local method=$2 n=$3
    local f splits d dj ckpt status=ok matcher=""
    f=$(factor_of "$scene")
    splits=$DATA/$scene/splits
    d=$RUNS/$scene/${method}_$n
    dj=$RUNS/$scene/${method}_${n}_joint
    ckpt=$d/gs/ckpts/ckpt_$((STOP_STEP - 1))_rank0.pt
    [[ $method == colmap ]] && matcher=exhaustive
    mkdir -p "$RUNS/$scene"
    exec 3>&1 > "$RUNS/$scene/${method}_$n.log" 2>&1
    echo "[$(date '+%F %T')] [START] $scene $method $n" | tee /dev/fd/3

    # 1. 训练帧位姿
    if [[ -e $d ]]; then
        echo "复用 $d"
    else
        (flock 9; timed "$d" pose estimate "$method" "$d" "$splits/train_$n.txt") 9> "$POSE_LOCK"
    fi
    has_model "$d" || status=failed:pose

    # 2. 联合运行（训练帧失败也照跑，便于看测试帧注册情况）
    if [[ -e $dj ]]; then
        echo "复用 $dj"
    else
        (flock 9; timed "$d" joint estimate "$method" "$dj" "$splits/joint_$n.txt") 9> "$POSE_LOCK"
    fi

    # 3. 训练
    if [[ $status == ok ]]; then
        if [[ -f $ckpt ]]; then
            echo "复用 $ckpt"
        else
            timed "$d" train train_gs "$d" "$f"
        fi
        [[ -f $ckpt ]] || status=failed:train
    fi

    # 4. 评测（联合运行对不齐时 eval.py 写 status=failed 的 JSON，append_result 记 failed:align）
    if [[ $status == ok ]]; then
        timed "$d" eval python "$REPO/scripts/eval.py" --result_dir "$d/gs" --ckpt "$ckpt" --joint_dir "$dj" \
            --test_list "$splits/test.txt" --out "$d/gs/eval.json" > "$d/eval.log" 2>&1 || status=failed:eval
    fi

    # 5. CSV
    python "$REPO/scripts/append_result.py" --csv "$RESULTS/$scene.csv" --scene "$scene" --method "$method" \
        --n "$n" --run_dir "$d" --train_list "$splits/train_$n.txt" --status "$status" --matcher "$matcher" \
        | tee /dev/fd/3
    echo "[$(date '+%F %T')] [END] $scene $method $n" | tee /dev/fd/3
}

if [[ ${1:-} == job ]]; then
    shift
    job "$@"
    exit 0
fi

SCENES=${SCENES:-desk lawn room garden}
METHODS=${METHODS:-colmap vggt}
NS=${NS:-8 16 32 all}
JOBS=${JOBS:-2}

# 先检查数据并生成联合运行帧列表（并行任务共用，不在 job 里写）
for scene in $SCENES; do
    for n in $NS; do
        s=$DATA/$scene/splits
        for x in "$s/train_$n.txt" "$s/test.txt"; do
            [[ -s $x ]] || { echo "缺少 $x（先跑 extract_frames.py）" >&2; exit 1; }
        done
        sort -u "$s/train_$n.txt" "$s/test.txt" > "$s/joint_$n.txt"
    done
done

echo "[$(date '+%F %T')] run_all: SCENES=($SCENES) METHODS=($METHODS) NS=($NS) JOBS=$JOBS STOP_STEP=$STOP_STEP"
for n in $NS; do
    for scene in $SCENES; do
        for method in $METHODS; do
            echo "$scene $method $n"
        done
    done
done | xargs -P "$JOBS" -L 1 "$0" job
echo "[$(date '+%F %T')] [ALL DONE]"
