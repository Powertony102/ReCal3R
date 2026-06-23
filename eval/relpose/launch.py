import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse
from datetime import timedelta

from copy import deepcopy
from eval.relpose.metadata import dataset_metadata

from add_ckpt_path import add_path_to_dust3r

from tqdm import tqdm
import time

RELPOSE_ALLOWED_DATASET_PREFIXES = ("scannet", "tum")
RELPOSE_ALLOWED_DATASETS = sorted(
    name
    for name in dataset_metadata.keys()
    if name.startswith(RELPOSE_ALLOWED_DATASET_PREFIXES)
)


def _device_from_any(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def run_inference_with_runtime_stats(views, model, device, inference_fn):
    device = _device_from_any(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = inference_fn(views, model, device)
        end_event.record()
        torch.cuda.synchronize(device)
        inference_time_ms = float(start_event.elapsed_time(end_event))
        peak_memory_allocated_mib = float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    else:
        start_time = time.perf_counter()
        outputs = inference_fn(views, model, device)
        inference_time_ms = float((time.perf_counter() - start_time) * 1000.0)
        peak_memory_allocated_mib = 0.0

    return outputs, inference_time_ms, peak_memory_allocated_mib


def save_runtime_metrics(output_dir, runtime_metrics):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "runtime_metrics.json"), "w") as f:
        json.dump(runtime_metrics, f, indent=4)


def summarize_runtime_metrics(save_dir):
    runtime_by_scene = {}
    for scene_name in sorted(os.listdir(save_dir)):
        scene_dir = os.path.join(save_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        runtime_metrics_path = os.path.join(scene_dir, "runtime_metrics.json")
        if not os.path.isfile(runtime_metrics_path):
            continue
        with open(runtime_metrics_path, "r") as f:
            runtime_by_scene[scene_name] = json.load(f)

    fps_scenes = {
        scene_name: float(metrics["fps"])
        for scene_name, metrics in runtime_by_scene.items()
        if "fps" in metrics
    }
    memory_scenes = {
        scene_name: float(metrics["peak_memory_allocated_mib"])
        for scene_name, metrics in runtime_by_scene.items()
        if "peak_memory_allocated_mib" in metrics
    }

    summary = {
        "fps": {
            "unit": "frames_per_second",
            "mean": float(np.mean(list(fps_scenes.values()))) if fps_scenes else 0.0,
            "scenes": fps_scenes,
        },
        "peak_memory_allocated_mib": {
            "unit": "MiB",
            "mean": float(np.mean(list(memory_scenes.values())))
            if memory_scenes
            else 0.0,
            "scenes": memory_scenes,
        },
    }

    with open(os.path.join(save_dir, "runtime_stats.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str.lower,
        default="scannet",
        choices=RELPOSE_ALLOWED_DATASETS,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="optional override for the selected dataset root",
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        choices=["cut3r", "ttt3r", "recal3r"],
        help="model type for state update strategy",
    )
    parser.add_argument(
        "--beta_base",
        type=float,
        default=None,
        help="safe fallback beta value; default is 0.1 for recal3r and 0.0 otherwise",
    )

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="maximum frames per sequence for pose evaluation",
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    parser.add_argument(
        "--dist_timeout_seconds",
        type=int,
        default=7200,
        help="timeout used to initialize torch distributed process group",
    )
    return parser

def resolve_beta_base_default(args):
    if args.beta_base is not None:
        return float(args.beta_base)
    return 0.1 if (args.model_update_type or "").strip() == "recal3r" else 0.0


def select_eval_filelist(filelist, max_frames):
    if max_frames is None or len(filelist) <= max_frames:
        return filelist
    if max_frames <= 1:
        return filelist[:max_frames]

    first_file = filelist[0]
    remaining_files = filelist[1:]
    step = max(1, len(remaining_files) // (max_frames - 1))
    return [first_file] + remaining_files[::step][: max_frames - 1]


def frame_ids_from_filelist(filelist):
    frame_ids = []
    for path in filelist:
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit():
            return None
        frame_ids.append(int(stem))
    return frame_ids


def timestamps_from_filelist(filelist):
    timestamps = []
    for path in filelist:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            timestamps.append(float(stem))
        except ValueError:
            return None
    return timestamps


def distributed_wait(distributed_state=None):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    backend = torch.distributed.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        device_idx = None
        if distributed_state is not None:
            local_idx = getattr(distributed_state, "local_process_index", None)
            if (
                isinstance(local_idx, int)
                and local_idx >= 0
                and local_idx < torch.cuda.device_count()
            ):
                device_idx = local_idx
        if device_idx is None:
            try:
                device_idx = torch.cuda.current_device()
            except Exception:
                device_idx = None
        if device_idx is not None:
            torch.distributed.barrier(device_ids=[device_idx])
            return

    torch.distributed.barrier()


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = args.dataset_path if args.dataset_path else metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter
    from accelerate import PartialState
    from eval.relpose.utils import (
        calculate_averages,
        eval_metrics,
        get_tum_poses,
        load_traj,
        plot_trajectory,
        process_directory,
        save_focals,
        save_intrinsics,
        save_tum_poses,
    )

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    if not os.path.isdir(img_path):
        raise FileNotFoundError(
            f"Dataset root does not exist: {img_path}. "
            "Check --dataset_path or the matching EVAL_*_ROOT environment variable."
        )

    seq_list = args.seq_list
    if seq_list is None:
        seq_list_func = metadata.get("seq_list_func", None)
        if seq_list_func is not None:
            seq_list = seq_list_func(img_path)
        else:
            if metadata.get("full_seq", False):
                args.full_seq = True
            if args.full_seq:
                seq_list = os.listdir(img_path)
                seq_list = [
                    seq
                    for seq in seq_list
                    if os.path.isdir(os.path.join(img_path, seq))
                ]
            else:
                seq_list = metadata.get("seq_list", [])
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    if not seq_list:
        raise RuntimeError(
            f"No sequences found for dataset '{args.eval_dataset}' under {img_path}. "
            "For raw TUM, the root should contain rgbd_dataset_* folders with "
            "rgb/, rgb.txt, and groundtruth.txt."
        )

    timeout_seconds = max(1, int(args.dist_timeout_seconds))
    try:
        distributed_state = PartialState(timeout=timedelta(seconds=timeout_seconds))
    except TypeError:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if (
            world_size > 1
            and torch.distributed.is_available()
            and not torch.distributed.is_initialized()
        ):
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=timeout_seconds),
            )
        distributed_state = PartialState()
        if distributed_state.is_main_process:
            print(
                "[WARN] accelerate.PartialState does not support timeout=... "
                "using torch.distributed.init_process_group(timeout=...) fallback."
            )
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

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
                if len(filelist) == 0:
                    raise RuntimeError(
                        f"No RGB frames found for sequence '{seq}' in {dir_path}."
                    )

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                    revisit=args.revisit,
                    update=not args.freeze_state,
                )

                (outputs, _), inference_time_ms, peak_memory_allocated_mib = (
                    run_inference_with_runtime_stats(
                        views, model, device, inference_recurrent_lighter
                    )
                )
                total_time_s = inference_time_ms / 1000.0
                fps = len(filelist) / total_time_s if total_time_s > 0 else 0.0
                print(
                    f"Finished pose estimation for {args.eval_dataset} {seq: <16}, "
                    f"FPS: {fps:.2f}, peak memory: {peak_memory_allocated_mib:.2f} MiB"
                )
                save_runtime_metrics(
                    f"{save_dir}/{seq}",
                    {
                        "num_frames": int(len(filelist)),
                        "inference_time_ms": float(inference_time_ms),
                        "fps": float(fps),
                        "peak_memory_allocated_mib": float(
                            peak_memory_allocated_mib
                        ),
                    },
                )

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(
                    outputs, revisit=args.revisit, solve_pose=args.solve_pose
                )

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                save_intrinsics(cam_dict, f"{save_dir}/{seq}/pred_intrinsics.txt")
                # save_depth_maps(pts3ds_self,f'{save_dir}/{seq}', conf_self=conf_self)
                # save_conf_maps(conf_self,f'{save_dir}/{seq}')
                # save_rgb_imgs(colors,f'{save_dir}/{seq}')

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)
                gt_frame_ids = None
                if traj_format == "scannet" and os.path.isdir(gt_traj_file):
                    gt_frame_ids = frame_ids_from_filelist(filelist)
                elif traj_format in ("bonn", "tum"):
                    gt_frame_ids = timestamps_from_filelist(filelist)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                        num_frames=len(filelist),
                        frame_ids=gt_frame_ids,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_wait(distributed_state)

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )
        runtime_summary = summarize_runtime_metrics(save_dir)
        print(
            f"Average FPS: {runtime_summary['fps']['mean']:.2f}, "
            f"Average peak memory: {runtime_summary['peak_memory_allocated_mib']['mean']:.2f} MiB"
        )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.beta_base = resolve_beta_base_default(args)
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import weighted_procrustes, geotrf

    args.full_seq = False
    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop, verbose=False)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1, solve_pose=False):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        if solve_pose:
            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self, 0)
        else:

            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pts3ds_self = torch.cat(pts3ds_self, 0)
            pr_poses = [
                pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
                for pred in outputs["pred"]
            ]
            pr_poses = torch.cat(pr_poses, 0)

            B, H, W, _ = pts3ds_self.shape
            pp = (
                torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
                .float()
                .repeat(B, 1)
                .reshape(B, 2)
            )
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = ARCroco3DStereo.from_pretrained(args.weights)
    
    # set model type
    model.model_update_type = args.model_update_type
    model.config.model_update_type = args.model_update_type
    model.config.beta_base = args.beta_base
    model.config.entropy_eps = 2e-14
    model.config.entropy_head_reduce = "mean"
    model.config.uncertainty_clamp_max = 1.0
    model.config.decay = 0.95 if args.model_update_type == "recal3r" else None
    model.beta_base = args.beta_base
    model.entropy_head_reduce = "mean"
    model.uncertainty_clamp_max = 1.0
    model.entropy_eps = 2e-14
    model.decay = 0.95 if args.model_update_type == "recal3r" else None

    eval_pose_estimation(args, model, save_dir=args.output_dir)
