import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
import time
from tqdm import tqdm

DEPTH_EVAL_ALIGN_CHOICES = ["metric", "scale", "scale&shift"]
DEPTH_ALLOWED_DATASET_PREFIXES = ("bonn", "tum")
DEPTH_ALLOWED_DATASETS = sorted(
    name
    for name in dataset_metadata.keys()
    if name.startswith(DEPTH_ALLOWED_DATASET_PREFIXES)
)
DEPTH_EVAL_DATASET_PREFIXES = DEPTH_ALLOWED_DATASET_PREFIXES


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
        type=str,
        default="tum",
        choices=DEPTH_ALLOWED_DATASETS,
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
        help="maximum frames per sequence for depth export",
    )
    parser.add_argument(
        "--depth_save_mode",
        type=str,
        default="full",
        choices=["full", "npy_only", "png_only", "none"],
        help=(
            "Depth export saving mode: full saves PNG+NPY; npy_only saves only "
            "NPY (recommended for eval); png_only saves only PNG; none saves neither."
        ),
    )
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
    parser.add_argument(
        "--run_depth_eval",
        action="store_true",
        help="run depth metric evaluation inside launch after depth export",
    )
    parser.add_argument(
        "--depth_eval_only",
        action="store_true",
        help="skip depth export and only run depth metric evaluation on existing predictions",
    )
    parser.add_argument(
        "--depth_eval_aligns",
        nargs="+",
        default=None,
        choices=DEPTH_EVAL_ALIGN_CHOICES,
        help="alignment modes for integrated depth evaluation",
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


def dataset_supports_depth_eval(eval_dataset):
    dataset = eval_dataset.lower()
    return dataset.startswith(DEPTH_EVAL_DATASET_PREFIXES)


def resolve_depth_eval_aligns(args):
    if args.depth_eval_aligns:
        return list(args.depth_eval_aligns)
    return list(DEPTH_EVAL_ALIGN_CHOICES)


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


def run_depth_eval_pipeline(args):
    from eval.video_depth import eval_depth

    if not dataset_supports_depth_eval(args.eval_dataset):
        print(
            f"[WARN] Dataset '{args.eval_dataset}' does not have integrated depth metric "
            "evaluation; skipping --run_depth_eval."
        )
        return

    align_modes = resolve_depth_eval_aligns(args)
    for align_mode in align_modes:
        print(f"[DepthEval] Running alignment mode: {align_mode}")
        eval_args = eval_depth.get_args_parser().parse_args([])
        eval_args.output_dir = args.output_dir
        eval_args.eval_dataset = args.eval_dataset
        eval_args.dataset_path = args.dataset_path if args.dataset_path else ""
        eval_args.pose_eval_stride = args.pose_eval_stride
        eval_args.max_frames = args.max_frames
        eval_args.seq_list = args.seq_list
        eval_args.align = align_mode
        eval_depth.main(eval_args)


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
        elif metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    if not seq_list:
        raise RuntimeError(
            f"No sequences found for dataset '{args.eval_dataset}' under {img_path}. "
            "For raw TUM, the root should contain rgbd_dataset_* folders with "
            "rgb/, depth/, rgb.txt, and depth.txt."
        )

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        assert load_img_size == 512
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
                )
                start = time.time()
                outputs, _ = inference_recurrent_lighter(views, model, device)
                end = time.time()
                fps = len(filelist) / (end - start)
                save_importance_log(model, f"{save_dir}/{seq}")
                save_entropy_log(model, f"{save_dir}/{seq}")

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(outputs)

                save_seq = metadata.get("save_seq_func", lambda seq: seq)(seq)
                os.makedirs(f"{save_dir}/{save_seq}", exist_ok=True)
                save_depth_maps(
                    pts3ds_self,
                    f"{save_dir}/{save_seq}",
                    conf_self=conf_self,
                    save_png=args.depth_save_mode in ("full", "png_only"),
                    save_npy=args.depth_save_mode in ("full", "npy_only"),
                )

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
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.beta_base = resolve_beta_base_default(args)
    args.full_seq = False
    args.no_crop = True

    distributed_state = PartialState()

    if not args.depth_eval_only:
        add_path_to_dust3r(args.weights)
        from dust3r.utils.image import load_images_for_eval as load_images
        from dust3r.post_process import estimate_focal_knowing_depth
        from dust3r.model import ARCroco3DStereo
        from dust3r.utils.camera import pose_encoding_to_camera

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
            images = load_images(img_paths, size=size, crop=crop)
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
                            else torch.from_numpy(
                                np.int32([raymaps[k].shape[1:-1][::-1]])
                            )
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
                        if r > 0 and not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                        new_views.append(new_view)
                return new_views
            return views

        def prepare_output(outputs, revisit=1):
            valid_length = len(outputs["pred"]) // revisit
            outputs["pred"] = outputs["pred"][-valid_length:]
            outputs["views"] = outputs["views"][-valid_length:]

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
            focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

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
    elif distributed_state.is_main_process:
        print("[DepthExport] Skipped because --depth_eval_only is enabled.")

    distributed_wait(distributed_state)

    if args.run_depth_eval and distributed_state.is_main_process:
        run_depth_eval_pipeline(args)
    distributed_wait(distributed_state)
