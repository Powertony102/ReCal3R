#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r

ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
scannet_root="${EVAL_SCANNET_ROOT:-/root/autodl-tmp/dataset/process_scannet}"
max_frames="${EVAL_SCANNET_MAX_FRAMES:-1000}"
max_frames_list="${EVAL_SCANNET_MAX_FRAMES_LIST:-$max_frames}"
num_processes="${NUM_PROCESSES:-2}"

for model_name in "${model_names[@]}"; do
    read -r -a max_frames_values <<< "${max_frames_list}"
    for max_frames in "${max_frames_values[@]}"; do
        output_dir="${workdir}/eval_results/relpose/scannet_${max_frames}/${model_name}"
        echo "$output_dir"
        accelerate launch --num_processes "$num_processes" --main_process_port 29550 eval/relpose/launch.py \
            --weights "$model_weights" \
            --output_dir "$output_dir" \
            --eval_dataset scannet \
            --dataset_path "$scannet_root" \
            --size 512 \
            --max_frames "$max_frames" \
            --model_update_type "$model_name"
    done
done
