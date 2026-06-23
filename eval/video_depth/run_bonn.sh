#!/bin/bash

set -e

workdir='.'
model_names=('recal3r') # recal3r ttt3r cut3r
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
dataset='bonn'
dataset_path="${EVAL_BONN_ROOT:-/root/autodl-tmp/dataset/rgbd_bonn_dataset}"
max_frames="${EVAL_BONN_MAX_FRAMES:-1000}"
max_frames_list="${EVAL_BONN_MAX_FRAMES_LIST:-$max_frames}"
pose_eval_stride="${EVAL_BONN_STRIDE:-1}"
depth_save_mode="${EVAL_BONN_DEPTH_SAVE_MODE:-full}"
num_processes="${EVAL_NUM_PROCESSES:-1}"
main_process_port="${EVAL_MAIN_PROCESS_PORT:-29556}"
alignments=('metric' 'scale' 'scale&shift')
# Optional: restrict sequences, e.g. EVAL_BONN_SEQ_LIST="balloon crowd2 person_tracking2".
seq_list_args=()
if [[ -n "${EVAL_BONN_SEQ_LIST:-}" ]]; then
    read -r -a seq_list_args <<< "${EVAL_BONN_SEQ_LIST}"
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
            --dataset_path "$dataset_path" \
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
