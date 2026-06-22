#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
dataset="${EVAL_TUM_DATASET:-tum}"
dataset_path="${EVAL_TUM_ROOT:-}"
max_frames="${EVAL_TUM_MAX_FRAMES:-1000}"
max_frames_list="${EVAL_TUM_MAX_FRAMES_LIST:-$max_frames}"
pose_eval_stride="${EVAL_TUM_STRIDE:-1}"
depth_save_mode="${EVAL_TUM_DEPTH_SAVE_MODE:-full}"
num_processes="${EVAL_NUM_PROCESSES:-1}"
main_process_port="${EVAL_MAIN_PROCESS_PORT:-29557}"
alignments=('metric' 'scale' 'scale&shift')

if [[ -z "${NUMEXPR_MAX_THREADS:-}" || ! "$NUMEXPR_MAX_THREADS" =~ ^[0-9]+$ || "$NUMEXPR_MAX_THREADS" -lt 128 ]]; then
    export NUMEXPR_MAX_THREADS=128
fi
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"

if [[ -n "$dataset_path" && ! -d "$dataset_path" ]]; then
    echo "TUM dataset root does not exist: $dataset_path" >&2
    alt_path="${dataset_path/autodl_tmp/autodl-tmp}"
    if [[ "$alt_path" != "$dataset_path" && -d "$alt_path" ]]; then
        echo "Did you mean: $alt_path ?" >&2
    fi
    exit 1
fi

dataset_path_args=()
if [[ -n "$dataset_path" ]]; then
    dataset_path_args=(--dataset_path "$dataset_path")
fi
seq_list_args=()
if [[ -n "${EVAL_TUM_SEQ_LIST:-}" ]]; then
    read -r -a seq_list_args <<< "${EVAL_TUM_SEQ_LIST}"
    seq_list_args=(--seq_list "${seq_list_args[@]}")
fi

for model_name in "${model_names[@]}"; do
    read -r -a max_frames_values <<< "${max_frames_list}"
    for max_frames in "${max_frames_values[@]}"; do
        output_dir="${workdir}/eval_results/video_depth/${dataset}_${max_frames}/${model_name}"
        echo "$output_dir"

        accelerate launch --num_processes "$num_processes" --main_process_port "$main_process_port" eval/video_depth/launch.py \
            --weights "$model_weights" \
            --output_dir "$output_dir" \
            --eval_dataset "$dataset" \
            "${dataset_path_args[@]}" \
            --pose_eval_stride "$pose_eval_stride" \
            --max_frames "$max_frames" \
            --depth_save_mode "$depth_save_mode" \
            --size 512 \
            --model_update_type "$model_name" \
            --run_depth_eval \
            --depth_eval_aligns "${alignments[@]}" \
            "${seq_list_args[@]}"
    done
done
