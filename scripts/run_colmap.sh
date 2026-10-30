#!/usr/bin/env bash
# COLMAP 4.x 稀疏重建（方法 A）：特征提取 → 匹配 → mapper，输出可直接给 gsplat 的数据目录。
#
# 用法：run_colmap.sh <图像目录> <输出目录> [帧列表.txt]
#   帧列表：每行一个文件名（相对图像目录）；不给则用目录下全部图像
#
# 环境变量：
#   MATCHER=sequential|exhaustive  默认 sequential（视频帧）；无序图像用 exhaustive
#   CAMERA_MODEL=PINHOLE           与 vggt_to_colmap.py 一致，gsplat 不用去畸变
#   USE_GPU=1                      COLMAP 4.2.1 源码编译带 CUDA（见 docs/setup.md）；CPU 版设 0
#   MIN_MODEL_SIZE=3               mapper 默认 10，少于 10 帧的子模型直接丢弃，N=8 时必然 0 注册
#
# 输出目录结构：
#   images/       指向输入图像的软链接（只含参与重建的帧）
#   database.db
#   sparse_all/   mapper 原始输出，可能有多个子模型
#   sparse/0      注册帧数最多的子模型（约定），gsplat 读这里
#   colmap.log    全部 COLMAP 输出
#   summary.tsv   表头 + 一行：注册数/总数、子模型数、各阶段耗时
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

img_dir=$(realpath "$1")
out=$2
list=${3:-}
MATCHER=${MATCHER:-sequential}
CAMERA_MODEL=${CAMERA_MODEL:-PINHOLE}
USE_GPU=${USE_GPU:-1}
MIN_MODEL_SIZE=${MIN_MODEL_SIZE:-3}

if [[ -e $out ]]; then
    echo "输出目录已存在：$out（删掉或换个名字）" >&2
    exit 1
fi
mkdir -p "$out/images" "$out/sparse_all"
out=$(realpath "$out")
log=$out/colmap.log

# 软链接出本次要用的帧；按文件名排序，sequential matcher 依赖这个顺序
if [[ -n $list ]]; then
    names=$(grep -v '^\s*$' "$list" | sort)
else
    names=$(ls "$img_dir" | grep -iE '\.(jpg|jpeg|png)$' | sort)
fi
while read -r n; do
    [[ -f $img_dir/$n ]] || { echo "找不到图像：$img_dir/$n" >&2; exit 1; }
    ln -s "$img_dir/$n" "$out/images/$n"
done <<< "$names"
n_input=$(wc -l <<< "$names" | tr -d ' ')

stage() {  # stage <名字> <命令...>：运行并把耗时（秒）存到 t_<名字>
    local name=$1; shift
    local t0=$SECONDS
    echo "== $name: $*" >> "$log"
    # mapper 失败（少帧时可能）也要写 summary；其他阶段失败直接退出
    "$@" >> "$log" 2>&1 || [[ $name == map ]] || { echo "$name 失败，见 $log" >&2; exit 1; }
    printf -v "t_$name" '%d' $((SECONDS - t0))
}

stage extract colmap feature_extractor \
    --database_path "$out/database.db" \
    --image_path "$out/images" \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu "$USE_GPU"

case $MATCHER in
    sequential) stage match colmap sequential_matcher \
        --database_path "$out/database.db" --FeatureMatching.use_gpu "$USE_GPU" ;;
    exhaustive) stage match colmap exhaustive_matcher \
        --database_path "$out/database.db" --FeatureMatching.use_gpu "$USE_GPU" ;;
    *) echo "未知 MATCHER：$MATCHER" >&2; exit 1 ;;
esac

stage map colmap mapper \
    --database_path "$out/database.db" \
    --image_path "$out/images" \
    --output_path "$out/sparse_all" \
    --Mapper.min_model_size "$MIN_MODEL_SIZE"

# 选注册帧数最多的子模型
best=""; n_reg=0; n_models=0
for m in "$out"/sparse_all/*/; do
    [[ -f $m/images.bin ]] || continue
    n_models=$((n_models + 1))
    r=$(colmap model_analyzer --path "$m" 2>&1 | sed -n 's/.*Registered images: *\([0-9]*\).*/\1/p')
    r=${r:-0}
    echo "子模型 $(basename "$m")：注册 $r 帧" >> "$log"
    if (( r > n_reg )); then n_reg=$r; best=$m; fi
done
if [[ -n $best ]]; then
    mkdir -p "$out/sparse"
    cp -r "$best" "$out/sparse/0"
fi

total=$((t_extract + t_match + t_map))
{
    printf 'out\tmatcher\tregistered\tinput\tmodels\textract_s\tmatch_s\tmap_s\ttotal_s\n'
    printf '%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' \
        "$out" "$MATCHER" "$n_reg" "$n_input" "$n_models" "$t_extract" "$t_match" "$t_map" "$total"
} > "$out/summary.tsv"
column -t -s $'\t' "$out/summary.tsv" 2>/dev/null || cat "$out/summary.tsv"
[[ -n $best ]] || { echo "mapper 没有产出任何模型，见 $log" >&2; exit 2; }
