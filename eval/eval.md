# Evaluation

## Datasets
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to download **ScanNet**, **TUM-dynamics**, **Sintel**, **Bonn**, **KITTI**, **7Scenes**, and **NRGBD** datasets.

### ScanNet
To prepare the **ScanNet** dataset, execute:
```bash
python datasets_preprocess/long_prepare_scannet.py # You may need to change the path of the dataset
```

### TUM-dynamics
To prepare the **TUM-dynamics** dataset, execute:
```bash
python datasets_preprocess/long_prepare_tum.py # You may need to change the path of the dataset
```
The relpose and video_depth evaluators can also read the original TUM-RGBD layout
directly: `ROOT/rgbd_dataset_*/rgb`, `ROOT/rgbd_dataset_*/depth`, `rgb.txt`,
`depth.txt`, and `groundtruth.txt`.

### Bonn
To prepare the **Bonn** dataset, execute:
```bash
python datasets_preprocess/long_prepare_bonn.py # You may need to change the path of the dataset
```
The relpose and mv_recon evaluators can also read the original Bonn layout directly:
`ROOT/rgbd_bonn_*/rgb`, `ROOT/rgbd_bonn_*/depth`, `rgb.txt`, `depth.txt`, and
`groundtruth.txt`.

### KITTI
To prepare the **KITTI** dataset, execute:
```bash
python datasets_preprocess/long_prepare_kitti.py # You may need to change the path of the dataset
```

### 7Scenes and NRGBD

The 3D reconstruction evaluator follows the 7Scenes/NRGBD layout used by [S-VGGT](https://github.com/Powertony102/S-VGGT):

- 7Scenes: `ROOT/<scene>/TrainSplit.txt`, `ROOT/<scene>/TestSplit.txt`, and `ROOT/<scene>/seq-xx/frame-000000.(color.png|depth.proj.png|pose.txt)`.
- NRGBD: `ROOT/<scene>/images/img0.png`, `ROOT/<scene>/depth/depth0.png`, and `ROOT/<scene>/poses.txt`.

You can pass roots explicitly with `--seven_scenes_root` and `--nrgbd_root`, or set `EVAL_7SCENES_ROOT` and `EVAL_NRGBD_ROOT`.

# Evaluation Scripts

Results will be saved in `eval_results/*`.

### Camera Pose Estimation

```bash
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_scannet.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('scannet_s3_1000')
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_tum.sh # Uses original TUM-RGBD layout by default; set EVAL_TUM_DATASET=tum_s1_1000 for prepared data
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_sintel.sh # You may need to change [--num_processes] to the number of your gpus
CUDA_VISIBLE_DEVICES=6,7 bash eval/relpose/run_7andN.sh # Uses EVAL_7SCENES_ROOT and EVAL_NRGBD_ROOT when set
```

For TUM-dynamics pose evaluation on the original dataset layout, pass the dataset
root that contains the `rgbd_dataset_*` sequence folders:

```bash
python eval/relpose/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir eval_results/relpose/tum/ttt3r \
  --eval_dataset tum \
  --dataset_path path/to/tum \
  --pose_eval_stride 1 \
  --max_frames 1000 \
  --size 512 \
  --model_update_type ttt3r
```

Use `--seq_list rgbd_dataset_freiburg3_walking_xyz` to evaluate only selected
TUM sequences. The helper script also accepts `EVAL_TUM_ROOT`, `EVAL_TUM_SEQ_LIST`,
`EVAL_TUM_MAX_FRAMES`, and `EVAL_TUM_STRIDE`.

For direct 7Scenes/NRGBD pose evaluation, pass the dataset root with `--dataset_path`:

```bash
python eval/relpose/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir eval_results/relpose/7scenes/ttt3r \
  --eval_dataset 7scenes \
  --dataset_path path/to/7-scenes \
  --pose_eval_stride 3 \
  --size 512 \
  --model_update_type ttt3r
```

For Bonn pose evaluation on the original dataset layout, pass the dataset root that
contains the `rgbd_bonn_*` sequence folders:

```bash
python eval/relpose/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir eval_results/relpose/bonn/ttt3r \
  --eval_dataset bonn \
  --dataset_path path/to/rgbd_bonn_dataset \
  --pose_eval_stride 1 \
  --max_frames 500 \
  --size 512 \
  --model_update_type ttt3r
```

Use `--seq_list balloon2 crowd2` to evaluate only selected Bonn sequences. Sequence
names may omit the `rgbd_bonn_` prefix.

### Video Depth

```bash
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_kitti.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('kitti_s1_500')
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_bonn.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('bonn_s1_500')
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_tum.sh # Uses original TUM-RGBD layout by default
CUDA_VISIBLE_DEVICES=5 bash eval/video_depth/run_sintel.sh # You may need to change [--num_processes] to the number of your gpus
```

For TUM-dynamics depth evaluation on the original dataset layout:

```bash
python eval/video_depth/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir eval_results/video_depth/tum/ttt3r \
  --eval_dataset tum \
  --dataset_path path/to/tum \
  --pose_eval_stride 1 \
  --max_frames 1000 \
  --size 512 \
  --model_update_type ttt3r

python eval/video_depth/eval_depth.py \
  --output_dir eval_results/video_depth/tum/ttt3r \
  --eval_dataset tum \
  --dataset_path path/to/tum \
  --pose_eval_stride 1 \
  --max_frames 1000 \
  --tum_depth_scale 5000 \
  --align scale\&shift
```

Use `--seq_list rgbd_dataset_freiburg3_walking_xyz` on both commands to evaluate
selected TUM sequences.

For Bonn depth evaluation on the original dataset layout:

```bash
python eval/video_depth/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir eval_results/video_depth/bonn/ttt3r \
  --eval_dataset bonn \
  --dataset_path path/to/rgbd_bonn_dataset \
  --pose_eval_stride 1 \
  --max_frames 1000 \
  --size 512 \
  --model_update_type ttt3r

python eval/video_depth/eval_depth.py \
  --output_dir eval_results/video_depth/bonn/ttt3r \
  --eval_dataset bonn \
  --dataset_path path/to/rgbd_bonn_dataset \
  --pose_eval_stride 1 \
  --max_frames 1000 \
  --align scale\&shift
```

Use `--seq_list balloon crowd2` on both commands to evaluate selected Bonn sequences.

