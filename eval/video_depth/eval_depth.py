import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.video_depth.tools import depth_evaluation, group_by_directory
import numpy as np
import cv2
from tqdm import tqdm
import glob
from PIL import Image
import argparse
import json
from pathlib import Path
from eval.video_depth.metadata import (
    bonn_depths_for_images,
    dataset_metadata,
    tum_depth_path_for_image,
)


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional override for dataset root.",
    )
    parser.add_argument(
        "--pose_eval_stride",
        default=1,
        type=int,
        help="stride used during depth export",
    )
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="maximum frames per sequence used during depth export",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences to evaluate",
    )
    parser.add_argument(
        "--bonn_depth_scale",
        type=float,
        default=5000.0,
        help="Bonn PNG depth scale; depth_m = depth_value / scale",
    )
    parser.add_argument(
        "--tum_depth_scale",
        type=float,
        default=5000.0,
        help="TUM-RGBD PNG depth scale; depth_m = depth_value / scale",
    )

    # Compatibility-only options. Depth metric evaluation consumes exported
    # frame_*.npy files, but sweep scripts may forward model inference flags.
    parser.add_argument("--weights", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, default=224, help=argparse.SUPPRESS)
    parser.add_argument("--no_crop", type=bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--beta_base",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    return parser


def select_eval_filelist(filelist, max_frames):
    if max_frames is None or len(filelist) <= max_frames:
        return filelist
    if max_frames <= 1:
        return filelist[:max_frames]

    first_file = filelist[0]
    remaining_files = filelist[1:]
    step = max(1, len(remaining_files) // (max_frames - 1))
    return [first_file] + remaining_files[::step][: max_frames - 1]


def main(args):
    if args.eval_dataset.startswith("bonn"):

        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array
            depth_png = np.asarray(Image.open(filename))
            depth = depth_png.astype(np.float64) / args.bonn_depth_scale
            depth[depth_png == 0] = -1.0
            return depth

        def get_video_results():
            gathered_depth_metrics = []
            metadata = dataset_metadata.get(args.eval_dataset)
            img_path = args.dataset_path if args.dataset_path else metadata["img_path"]
            seq_list = args.seq_list
            if seq_list is None:
                seq_list_func = metadata.get("seq_list_func", None)
                if seq_list_func is not None:
                    seq_list = seq_list_func(img_path)
                else:
                    seq_list = metadata.get("seq_list", [])
            seq_list = sorted(seq_list)

            for seq in tqdm(seq_list):
                dir_path = metadata["dir_path_func"](img_path, seq)
                filelist_func = metadata.get("filelist_func", None)
                if filelist_func is None:
                    filelist = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                    filelist.sort()
                else:
                    filelist = filelist_func(dir_path)
                filelist = filelist[:: args.pose_eval_stride]
                filelist = select_eval_filelist(filelist, args.max_frames)

                gt_pathes = bonn_depths_for_images(filelist)
                save_seq = seq.removeprefix("rgbd_bonn_")
                pred_dir = os.path.join(args.output_dir, save_seq)
                if not os.path.isdir(pred_dir):
                    pred_dir = os.path.join(args.output_dir, seq)
                pd_pathes = sorted(glob.glob(os.path.join(pred_dir, "frame_*.npy")))

                pair_count = min(len(pd_pathes), len(gt_pathes))
                if pair_count == 0:
                    print(f"Warning: no Bonn depth pairs found for {seq}, skipping.")
                    continue
                if len(pd_pathes) != len(gt_pathes):
                    print(
                        f"Warning: {seq} has {len(pd_pathes)} predictions and "
                        f"{len(gt_pathes)} GT depths; evaluating first {pair_count} pairs."
                    )
                pd_pathes = pd_pathes[:pair_count]
                gt_pathes = gt_pathes[:pair_count]

                valid_pd_pathes = []
                gt_depth_list = []
                for pd_path, gt_path in zip(pd_pathes, gt_pathes):
                    gt_depth_i = depth_read(gt_path)
                    if not np.any(gt_depth_i > 0):
                        print(f"Warning: skipping invalid Bonn depth frame {gt_path}")
                        continue
                    valid_pd_pathes.append(pd_path)
                    gt_depth_list.append(gt_depth_i)

                if len(gt_depth_list) == 0:
                    print(f"Warning: no valid Bonn GT depth frames for {seq}, skipping.")
                    continue

                pd_pathes = valid_pd_pathes
                gt_depth = np.stack(gt_depth_list, axis=0)
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=True,
                        )
                    )
                gathered_depth_metrics.append(depth_results)

                # seq_len = gt_depth.shape[0]
                # error_map = error_map.reshape(seq_len, -1, error_map.shape[-1]).cpu()
                # error_map_colored = colorize(error_map, range=(error_map.min(), error_map.max()), append_cbar=True)
                # ImageSequenceClip([x for x in (error_map_colored.numpy()*255).astype(np.uint8)], fps=10).write_videofile(f'{args.output_dir}/errormap_{key}_{args.align}.mp4', fps=10)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            if len(gathered_depth_metrics) == 0:
                print("Warning: No depth metrics were gathered.")
                average_metrics = {}
            else:
                average_metrics = {
                    key: np.average(
                        [metrics[key] for metrics in gathered_depth_metrics],
                        weights=[
                            metrics["valid_pixels"] for metrics in gathered_depth_metrics
                        ],
                    )
                    for key in gathered_depth_metrics[0].keys()
                    if key != "valid_pixels"
                }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset.startswith("tum"):

        def depth_read(filename):
            depth_png = np.asarray(Image.open(filename))
            depth = depth_png.astype(np.float64) / args.tum_depth_scale
            depth[depth_png == 0] = -1.0
            return depth

        def get_video_results():
            gathered_depth_metrics = []
            metadata = dataset_metadata.get(args.eval_dataset)
            img_path = args.dataset_path if args.dataset_path else metadata["img_path"]
            seq_list = args.seq_list
            if seq_list is None:
                seq_list_func = metadata.get("seq_list_func", None)
                if seq_list_func is not None:
                    seq_list = seq_list_func(img_path)
                else:
                    seq_list = metadata.get("seq_list", [])
            seq_list = sorted(seq_list)

            for seq in tqdm(seq_list):
                dir_path = metadata["dir_path_func"](img_path, seq)
                filelist_func = metadata.get("filelist_func", None)
                if filelist_func is None:
                    filelist = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                    filelist.sort()
                else:
                    filelist = filelist_func(dir_path)
                filelist = filelist[:: args.pose_eval_stride]
                filelist = select_eval_filelist(filelist, args.max_frames)

                save_seq = metadata.get("save_seq_func", lambda seq: seq)(seq)
                pred_dir = os.path.join(args.output_dir, save_seq)
                if not os.path.isdir(pred_dir):
                    pred_dir = os.path.join(args.output_dir, seq)
                pd_pathes = sorted(glob.glob(os.path.join(pred_dir, "frame_*.npy")))

                if len(pd_pathes) != len(filelist):
                    print(
                        f"Warning: {seq} has {len(pd_pathes)} predictions and "
                        f"{len(filelist)} selected RGB frames; pairing by frame index."
                    )

                valid_pd_pathes = []
                gt_depth_list = []
                for frame_idx, image_path in enumerate(filelist):
                    if frame_idx >= len(pd_pathes):
                        break
                    gt_path = tum_depth_path_for_image(image_path)
                    if gt_path is None:
                        print(f"Warning: no TUM depth match for {image_path}")
                        continue
                    gt_depth_i = depth_read(gt_path)
                    if not np.any(gt_depth_i > 0):
                        print(f"Warning: skipping invalid TUM depth frame {gt_path}")
                        continue
                    valid_pd_pathes.append(pd_pathes[frame_idx])
                    gt_depth_list.append(gt_depth_i)

                if len(gt_depth_list) == 0:
                    print(f"Warning: no valid TUM GT depth frames for {seq}, skipping.")
                    continue

                pd_pathes = valid_pd_pathes
                gt_depth = np.stack(gt_depth_list, axis=0)
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in pd_pathes
                    ],
                    axis=0,
                )

                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_lad2=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            align_with_scale=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=70,
                            metric_scale=True,
                            use_gpu=True,
                        )
                    )
                gathered_depth_metrics.append(depth_results)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            if len(gathered_depth_metrics) == 0:
                print("Warning: No depth metrics were gathered.")
                average_metrics = {}
            else:
                average_metrics = {
                    key: np.average(
                        [metrics[key] for metrics in gathered_depth_metrics],
                        weights=[
                            metrics["valid_pixels"] for metrics in gathered_depth_metrics
                        ],
                    )
                    for key in gathered_depth_metrics[0].keys()
                    if key != "valid_pixels"
                }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    elif args.eval_dataset.startswith("kitti"):

        def depth_read(filename):
            # loads depth map D from png file
            # and returns it as a numpy array,
            # for details see readme.txt
            img_pil = Image.open(filename)
            depth_png = np.array(img_pil, dtype=int)
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255

            depth = depth_png.astype(float) / 256.0
            depth[depth_png == 0] = -1.0
            return depth

        # extract number from dataset name, e.g. kitti_100 -> 100
        if "_" in args.eval_dataset:
            kitti_number = args.eval_dataset.split("_")[-1]
        else:
            kitti_number = "110"  # default value
        
        default_kitti_root = Path(
            "./data/long_kitti_s1/depth_selection/val_selection_cropped"
        )
        kitti_root = default_kitti_root
        if args.dataset_path:
            dataset_path_obj = Path(args.dataset_path).expanduser().resolve()
            if dataset_path_obj.name.startswith("image_gathered_"):
                kitti_root = dataset_path_obj.parent

        gt_root = kitti_root / f"groundtruth_depth_gathered_{kitti_number}"
        depth_pathes = glob.glob(str(gt_root / "*/*.png"))
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/frame_*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
        print(f"KITTI GT root: {gt_root}")
        print(f"Found KITTI GT files: {len(depth_pathes)}")
        print(f"Found KITTI prediction files: {len(pred_pathes)}")
        if len(depth_pathes) == 0:
            raise RuntimeError(
                f"No KITTI GT depth files found under {gt_root}. "
                "Check long_prepare_kitti.py outputs and --dataset_path."
            )
        if len(pred_pathes) == 0:
            raise RuntimeError(
                f"No prediction files found under {args.output_dir}/*/frame_*.npy. "
                "Check depth export stage and --depth_save_mode."
            )

        def get_video_results():
            grouped_pred_depth = group_by_directory(pred_pathes)
            grouped_gt_depth = group_by_directory(depth_pathes)
            print(
                f"KITTI grouped keys: pred={len(grouped_pred_depth)}, gt={len(grouped_gt_depth)}"
            )
            gathered_depth_metrics = []
            for key in tqdm(grouped_pred_depth.keys()):
                pd_pathes = grouped_pred_depth[key]
                gt_pathes = grouped_gt_depth.get(key, [])
                if len(gt_pathes) == 0:
                    print(f"Warning: no KITTI GT depth found for sequence '{key}', skipping.")
                    continue

                pair_count = min(len(pd_pathes), len(gt_pathes))
                if pair_count == 0:
                    print(f"Warning: no valid KITTI pairs for sequence '{key}', skipping.")
                    continue
                if len(pd_pathes) != len(gt_pathes):
                    print(
                        f"Warning: {key} has {len(pd_pathes)} predictions and "
                        f"{len(gt_pathes)} GT depths; evaluating first {pair_count} pairs."
                    )
                pd_pathes = pd_pathes[:pair_count]
                gt_pathes = gt_pathes[:pair_count]

                valid_pd_pathes = []
                gt_depth_list = []
                for pd_path, gt_path in zip(pd_pathes, gt_pathes):
                    gt_depth_i = depth_read(gt_path)
                    if not np.any(gt_depth_i > 0):
                        print(f"Warning: skipping invalid KITTI depth frame {gt_path}")
                        continue
                    valid_pd_pathes.append(pd_path)
                    gt_depth_list.append(gt_depth_i)

                if len(gt_depth_list) == 0:
                    print(f"Warning: no valid KITTI GT depth frames for {key}, skipping.")
                    continue

                gt_depth = np.stack(gt_depth_list, axis=0)
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in valid_pd_pathes
                    ],
                    axis=0,
                )

                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                if args.align == "scale&shift":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_lad2=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "scale":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            align_with_scale=True,
                            use_gpu=True,
                        )
                    )
                elif args.align == "metric":
                    depth_results, error_map, depth_predict, depth_gt = (
                        depth_evaluation(
                            pr_depth,
                            gt_depth,
                            max_depth=None,
                            metric_scale=True,
                            use_gpu=True,
                        )
                    )
                gathered_depth_metrics.append(depth_results)

            depth_log_path = f"{args.output_dir}/result_{args.align}.json"
            if len(gathered_depth_metrics) == 0:
                raise RuntimeError(
                    "No KITTI depth metrics were gathered after sequence matching. "
                    "Check sequence folder names between predictions and GT."
                )
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                )
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            print("Average depth evaluation metrics:", average_metrics)
            with open(depth_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
