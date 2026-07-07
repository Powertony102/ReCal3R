#!/bin/bash

set -e

workdir='.'
model_names=('recal3r') # recal3r ttt3r cut3r
eval_datasets=('7scenes' 'nrgbd')
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
kf=2
max_eval_points=1000000
seven_scenes_root="${EVAL_7SCENES_ROOT:-data/7-scenes}"
nrgbd_root="${EVAL_NRGBD_ROOT:-data/nrgbd}"

for model_name in "${model_names[@]}"; do
for data in "${eval_datasets[@]}"; do

# for max_frames in 50 100 150 200 250 300 350 400
for max_frames in 200

do
    output_dir="${workdir}/eval_results/video_recon/${data}_${max_frames}/${model_name}"
    echo "$output_dir"
    NCCL_TIMEOUT=360000 accelerate launch --num_processes 1 --main_process_port 29502 eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --model_update_type "$model_name" \
        --eval_dataset "$data" \
        --seven_scenes_root "$seven_scenes_root" \
        --nrgbd_root "$nrgbd_root" \
        --kf "$kf" \
        --max_frames "$max_frames" \
        --max_eval_points "$max_eval_points"

done
done
done
