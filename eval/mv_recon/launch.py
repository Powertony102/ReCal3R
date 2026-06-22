import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import os.path as osp
import json
import io
from pathlib import Path
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--eval_dataset",
        type=str.lower,
        default="7scenes",
        choices=["7scenes", "nrgbd"],
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="override root for the selected single dataset",
    )
    parser.add_argument(
        "--seven_scenes_root",
        type=str,
        default=os.environ.get(
            "EVAL_7SCENES_ROOT", "/root/autodl-tmp/dataset/7-scenes"
        ),
        help="7Scenes dataset root",
    )
    parser.add_argument(
        "--nrgbd_root",
        type=str,
        default=os.environ.get(
            "EVAL_NRGBD_ROOT", "/root/autodl-tmp/dataset/nrgbd"
        ),
        help="NRGBD dataset root",
    )
    parser.add_argument(
        "--scannet_root",
        type=str,
        default=os.environ.get(
            "EVAL_SCANNET_ROOT",
            "/home/jovyan/shared/xinzeli/scannetv2/process_scannet",
        ),
        help="processed ScanNet root with scene/color and scene/pose folders",
    )
    parser.add_argument(
        "--scannet_gt_ply_dir",
        "--gt_ply_dir",
        dest="scannet_gt_ply_dir",
        type=str,
        default=os.environ.get(
            "EVAL_SCANNET_GT_PLY_DIR", "/home/jovyan/shared/xinzeli/scannetv2/scannet"
        ),
        help="ScanNet ground-truth mesh root for reconstruction Chamfer evaluation",
    )
    parser.add_argument(
        "--bonn_root",
        type=str,
        default=os.environ.get("EVAL_BONN_ROOT", "data/bonn/rgbd_bonn_dataset"),
        help="Bonn RGB-D dataset root containing rgbd_bonn_* sequence folders",
    )
    parser.add_argument(
        "--bonn_depth_scale",
        type=float,
        default=5000.0,
        help="Bonn PNG depth scale; depth_m = depth_value / scale",
    )
    parser.add_argument(
        "--bonn_intrinsics",
        type=float,
        nargs=4,
        default=None,
        metavar=("FX", "FY", "CX", "CY"),
        help="override Bonn intrinsics as fx fy cx cy",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="maximum ScanNet scenes to evaluate",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="maximum distance clipping for ScanNet Chamfer Distance",
    )
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--kf",
        "--kf_every",
        dest="kf_every",
        type=int,
        default=2,
        help="sample one keyframe every N frames",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        choices=["cut3r", "ttt3r", "recal3r"],
        help="model update type",
    )
    parser.add_argument(
        "--beta_safe",
        type=float,
        default=None,
        help="safe fallback beta value; default is 0.1 for recal3r and 0.0 otherwise",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="voxel size for voxel grid downsampling, 0 means no downsampling",
    )
    parser.add_argument(
        "--max_eval_points",
        type=int,
        default=1_000_000,
        help="randomly subsample each point cloud to at most this many points for evaluation",
    )
    return parser

def resolve_beta_safe_default(args):
    if args.beta_safe is not None:
        return float(args.beta_safe)
    return 0.1 if (args.model_update_type or "").strip() == "recal3r" else 0.0


def summarize_scannet_metrics(save_path, input_frame):
    metric_names = [
        "chamfer_distance",
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "inference_time_ms",
        "fps",
    ]
    metrics_by_scene = {}
    input_frame_dir = Path(save_path) / f"input_frame_{input_frame}"
    for metrics_path in sorted(input_frame_dir.glob("*/metrics.json")):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics_by_scene[metrics_path.parent.name] = {
            key: float(metrics[key]) for key in metric_names if key in metrics
        }

    averages = {
        key: float(
            np.mean(
                [
                    scene_metrics[key]
                    for scene_metrics in metrics_by_scene.values()
                    if key in scene_metrics
                ]
            )
        )
        if any(key in scene_metrics for scene_metrics in metrics_by_scene.values())
        else 0.0
        for key in metric_names
    }

    input_frame_dir.mkdir(parents=True, exist_ok=True)
    with open(input_frame_dir / "all_scenes_metrics.json", "w") as f:
        json.dump({"scenes": metrics_by_scene, "average": averages}, f, indent=4)
    with open(input_frame_dir / "average_metrics.json", "w") as f:
        json.dump(averages, f, indent=4)

    print("\nScanNet average metrics:")
    for metric_name, value in averages.items():
        print(f"{metric_name}: {value:.6f}")
    return averages


def umeyama_alignment(src, dst, estimate_scale=True):
    from scipy.linalg import svd

    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered @ src_centered.T

    U, D, Vt = svd(cov)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(U @ V.T) < 0:
        S[2, 2] = -1

    R = U @ S @ V.T
    if estimate_scale:
        src_var = np.sum(src_centered * src_centered)
        scale = 1.0 if src_var < 1e-10 else np.sum(D * np.diag(S)) / src_var
    else:
        scale = 1.0
    t = dst_mean.ravel() - scale * (R @ src_mean).ravel()
    return scale, R, t


def align_point_clouds_scale(source_pc, target_pc):
    source_min = np.min(source_pc, axis=0)
    source_max = np.max(source_pc, axis=0)
    target_min = np.min(target_pc, axis=0)
    target_max = np.max(target_pc, axis=0)

    source_center = (source_max + source_min) / 2
    target_center = (target_max + target_min) / 2
    source_diag = np.linalg.norm(source_max - source_min)
    target_diag = np.linalg.norm(target_max - target_min)
    scale = 1.0 if source_diag < 1e-8 else target_diag / source_diag
    return (source_pc - source_center) * scale + target_center, scale


def compute_scannet_chamfer(points_pred, points_gt, max_dist=1.0):
    import open3d as o3d

    max_points = 100000
    if points_pred.shape[0] > max_points:
        indices = np.random.choice(points_pred.shape[0], max_points, replace=False)
        points_pred = points_pred[indices]
    if points_gt.shape[0] > max_points:
        indices = np.random.choice(points_gt.shape[0], max_points, replace=False)
        points_gt = points_gt[indices]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
    pcd_pred = pcd_pred.voxel_down_sample(0.05)
    pcd_gt = pcd_gt.voxel_down_sample(0.05)

    distances_pred = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    distances_gt = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    distances_pred = np.clip(distances_pred, 0, max_dist)
    distances_gt = np.clip(distances_gt, 0, max_dist)
    return float(np.mean(distances_pred) + np.mean(distances_gt))


def load_scannet_gt_pointcloud(scene_id, gt_ply_dir):
    import open3d as o3d

    ply_path = Path(gt_ply_dir) / scene_id / f"{scene_id}_vh_clean_2.ply"
    if not ply_path.exists():
        print(f"Warning: ScanNet GT mesh not found: {ply_path}")
        return None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        print(f"Warning: ScanNet GT mesh has no points: {ply_path}")
        return None
    return points


def eval_scannet_trajectory(poses_est, poses_gt, frame_ids, plot_flag=False):
    import matplotlib.pyplot as plt
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    import evo.tools.plot as evo_plot
    from PIL import Image
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D
    from scipy.spatial.transform import Rotation

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(
            scalar_first=True
        ),
        timestamps=np.array(frame_ids[: len(poses_gt)], dtype=float),
    )
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(
            scalar_first=True
        ),
        timestamps=np.array(frame_ids[: len(poses_est)], dtype=float),
    )

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        align_origin=True,
    )
    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        align_origin=True,
    )
    rpe_rot_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_trans_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )

    metrics = {
        "ate": float(ate_result.stats["rmse"]),
        "are": float(are_result.stats["rmse"]),
        "rpe_rot": float(rpe_rot_result.stats["rmse"]),
        "rpe_trans": float(rpe_trans_result.stats["rmse"]),
    }

    traj_plot = None
    if plot_flag:
        fig = plt.figure()
        ax = evo_plot.prepare_axis(fig, evo_plot.PlotMode.xz)
        ax.set_title(f"ATE: {metrics['ate']:.3f}, ARE: {metrics['are']:.3f}")
        evo_plot.traj(ax, evo_plot.PlotMode.xz, traj_ref, "--", "gray", "gt")
        evo_plot.traj_colormap(
            ax,
            ate_result.trajectories["traj"],
            ate_result.np_arrays["error_array"],
            evo_plot.PlotMode.xz,
            min_map=ate_result.stats["min"],
            max_map=ate_result.stats["max"],
        )
        ax.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=90)
        buffer.seek(0)
        traj_plot = Image.open(buffer)
        traj_plot.load()
        buffer.close()
        plt.close(fig)

    return metrics, traj_plot


def evaluate_scannet_scene_and_save(
    scene_id,
    c2ws,
    first_gt_pose,
    frame_ids,
    trajectory_poses,
    merged_points,
    output_scene_dir,
    gt_ply_dir,
    chamfer_max_dist,
    inference_time_ms,
    plot_flag,
):
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(trajectory_poses), len(c2ws))
    metrics, traj_plot = eval_scannet_trajectory(
        np.asarray(trajectory_poses[:n]),
        np.linalg.inv(c2ws[:n]),
        frame_ids[:n],
        plot_flag=plot_flag,
    )

    if merged_points.shape[0] > 0:
        gt_points = load_scannet_gt_pointcloud(scene_id, gt_ply_dir)
        if gt_points is not None:
            homogeneous_points = np.hstack(
                [merged_points, np.ones((merged_points.shape[0], 1))]
            )
            world_points_raw = (homogeneous_points @ first_gt_pose.T)[:, :3]
            world_points_scaled, scale_factor = align_point_clouds_scale(
                world_points_raw, gt_points
            )
            metrics["chamfer_distance"] = compute_scannet_chamfer(
                world_points_scaled, gt_points, max_dist=chamfer_max_dist
            )
            metrics["scale_factor"] = float(scale_factor)

    for metric_name, metric_value in list(metrics.items()):
        metrics[f"aligned_{metric_name}"] = metric_value
    metrics["inference_time_ms"] = float(inference_time_ms)
    total_time_s = float(inference_time_ms) / 1000.0
    metrics["fps"] = float(n / total_time_s) if total_time_s > 0 else 0.0

    with open(output_scene_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    if traj_plot is not None:
        traj_plot.save(output_scene_dir / "plot.png")

    return metrics


def evaluate_scannet_cut3r_scene(
    args,
    batch,
    preds,
    save_path,
    scene_id,
    inference_time_ms,
    fps,
):
    c2ws = np.stack(
        [view["scannet_camera_pose"][0].detach().cpu().numpy() for view in batch]
    )
    first_gt_pose = batch[0]["scannet_first_gt_pose"][0].detach().cpu().numpy()
    frame_ids = [int(view["scannet_frame_id"][0]) for view in batch]

    all_world_points = []
    for view_idx, pred in enumerate(preds):
        pts = pred["pts3d_in_other_view"].to(torch.float32).detach().cpu().numpy()[0]
        conf = pred.get("conf", None)
        if conf is not None and args.conf_thresh > 0:
            conf_mask = (
                conf.to(torch.float32).detach().cpu().numpy()[0] > args.conf_thresh
            )
        else:
            conf_mask = np.ones(pts.shape[:2], dtype=bool)

        pts = pts.reshape(-1, 3)
        conf_mask = conf_mask.reshape(-1)
        valid_mask = np.isfinite(pts).all(axis=1) & conf_mask
        pts = pts[valid_mask]
        if pts.shape[0] == 0:
            continue

        c2w = c2ws[view_idx]
        pts_world = (c2w[:3, :3] @ pts.T).T + c2w[:3, 3]
        all_world_points.append(pts_world)

    if len(all_world_points) == 0:
        print(f"Skipping {scene_id}: no valid predicted points")
        return None

    merged_points = np.vstack(all_world_points)
    if args.max_eval_points is not None and args.max_eval_points > 0:
        if merged_points.shape[0] > args.max_eval_points:
            sample_indices = np.random.choice(
                merged_points.shape[0], args.max_eval_points, replace=False
            )
            merged_points = merged_points[sample_indices]

    input_frame = args.max_frames if args.max_frames is not None else len(batch)
    output_scene_dir = Path(save_path) / f"input_frame_{input_frame}" / scene_id
    return evaluate_scannet_scene_and_save(
        scene_id,
        c2ws,
        first_gt_pose,
        frame_ids,
        [c2ws[i] for i in range(len(c2ws))],
        merged_points,
        output_scene_dir,
        Path(args.scannet_gt_ply_dir),
        args.chamfer_max_dist,
        inference_time_ms,
        args.plot,
    )


def build_datasets(args, resolution):
    from eval.mv_recon.data import SevenScenes, NRGBD

    selected_dataset = args.eval_dataset.lower()

    datasets_all = {}
    if selected_dataset == "7scenes":
        root = (
            args.dataset_root
            if selected_dataset == "7scenes" and args.dataset_root
            else args.seven_scenes_root
        )
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
        )

    if selected_dataset == "nrgbd":
        root = (
            args.dataset_root
            if selected_dataset == "nrgbd" and args.dataset_root
            else args.nrgbd_root
        )
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
        )

    return datasets_all


def sampled_view_count(frame_count, kf_every, max_frames=None):
    from eval.mv_recon.data import sample_frame_ids

    frame_ids = list(range(frame_count))
    return len(sample_frame_ids(frame_ids, kf_every, max_frames=max_frames))


def main(args):
    import open3d as o3d
    from accelerate import Accelerator

    add_path_to_dust3r(args.weights)
    from eval.mv_recon.utils import accuracy, completion, chamfer_distance

    # Keep point subsampling reproducible across runs.
    np.random.seed(0)

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = build_datasets(args, resolution)

    # ====== print the number of views for each scene ======
    print("\n=== number of views for each scene ===")
    for name_data, dataset in datasets_all.items():
        print(f"\n{name_data} dataset:")
        for scene_id in dataset.scene_list:
            if name_data.lower() == "nrgbd":
                # NRGBD dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id, "images")
                num_files = len(
                    [name for name in os.listdir(data_path) if name.endswith(".png")]
                )
            elif name_data.lower() == "scannet":
                data_path = osp.join(dataset.ROOT, scene_id, "color")
                num_files = len(
                    [
                        name
                        for name in os.listdir(data_path)
                        if name.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
            elif name_data.lower() == "bonn":
                from eval.mv_recon.data import _load_bonn_records

                data_path = osp.join(dataset.ROOT, scene_id)
                num_files = len(
                    _load_bonn_records(data_path, max_time_diff=dataset.max_time_diff)
                )
            else:
                # SevenScenes dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id)
                num_files = len(
                    [name for name in os.listdir(data_path) if "color" in name]
                )

            view_count = sampled_view_count(
                num_files, dataset.kf_every, max_frames=dataset.max_frames
            )
            original_view_count = sampled_view_count(num_files, dataset.kf_every)
            if dataset.max_frames is not None and view_count != original_view_count:
                print(
                    f"  {scene_id}: {view_count} views "
                    f"(original: {original_view_count}, limit: {dataset.max_frames})"
                )
            else:
                print(f"  {scene_id}: {view_count} views")
    print("================================\n")
    # ====== print end ======

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    # if model_name == "ours" or model_name == "cut3r":
    from dust3r.model import ARCroco3DStereo
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
    model.model_update_type = args.model_update_type
    model.config.model_update_type = args.model_update_type
    model.config.beta_safe = args.beta_safe
    model.config.entropy_eps = 2e-14
    model.config.entropy_head_reduce = "mean"
    model.config.uncertainty_clamp_max = 1.0
    model.config.decay = 0.95 if args.model_update_type == "recal3r" else None
    model.beta_safe = args.beta_safe
    model.entropy_head_reduce = "mean"
    model.uncertainty_clamp_max = 1.0
    model.entropy_eps = 2e-14
    model.decay = 0.95 if args.model_update_type == "recal3r" else None

    model.eval()
    # else:
    #     raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            chamfer_all = 0
            chamfer_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                            "scannet_frame_id",
                            "scannet_camera_pose",
                            "scannet_first_gt_pose",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    # if model_name == "ours" or model_name == "cut3r":
                    revisit = args.revisit
                    update = not args.freeze
                    if revisit > 1:
                        # repeat input for 'revisit' times
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i)
                                    for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i)
                                    for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0:
                                    if not update:
                                        new_view["update"] = torch.zeros_like(
                                            batch[i]["update"]
                                        ).bool()
                                new_views.append(new_view)
                        batch = new_views
                    with torch.amp.autocast('cuda', enabled=False):
                        start = time.time()
                        output = model(batch)
                        # preds, batch = model.forward_recurrent_light(batch)
                        end = time.time()
                        preds, batch = output.ress, output.views
                    valid_length = len(preds) // revisit
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]
                    fps = len(batch) / (end - start)
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                    )
                    fps_all.append(fps)
                    time_all.append(end - start)

                    if name_data.lower() == "scannet":
                        scene_id = batch[0]["label"][0].rsplit("/", 1)[0]
                        metrics = evaluate_scannet_cut3r_scene(
                            args,
                            batch,
                            preds,
                            save_path,
                            scene_id,
                            float((end - start) * 1000.0),
                            float(fps),
                        )
                        if metrics is not None:
                            print(
                                f"Idx: {scene_id}, Chamfer: {metrics.get('chamfer_distance', 0.0)}, "
                                f"ATE: {metrics.get('ate', 0.0)}, RPE trans: {metrics.get('rpe_trans', 0.0)}, "
                                f"RPE rot: {metrics.get('rpe_rot', 0.0)}, FPS: {fps}",
                                file=open(log_file, "a"),
                            )
                            print(
                                f"Idx: {scene_id}, Chamfer: {metrics.get('chamfer_distance', 0.0)}, "
                                f"ATE: {metrics.get('ate', 0.0)}, RPE trans: {metrics.get('rpe_trans', 0.0)}, "
                                f"RPE rot: {metrics.get('rpe_rot', 0.0)}, FPS: {fps}"
                            )
                        torch.cuda.empty_cache()
                        continue

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]
                        # mask = mask & (conf > 1.8)

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        #### Align predicted 3D points to the ground truth
                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)

                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    pred_finite_mask = np.isfinite(pts_all_masked).all(axis=-1)
                    pts_all_masked = pts_all_masked[pred_finite_mask]
                    images_all_masked = images_all_masked[pred_finite_mask]

                    gt_finite_mask = np.isfinite(pts_gt_all_masked).all(axis=-1)
                    pts_gt_all_masked = pts_gt_all_masked[gt_finite_mask]

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    images_all_masked = images_all_masked.reshape(-1, 3)

                    max_eval_points = args.max_eval_points
                    if max_eval_points is not None and max_eval_points > 0:
                        if pts_all_masked.shape[0] > max_eval_points:
                            sample_indices = np.random.choice(
                                pts_all_masked.shape[0], max_eval_points, replace=False
                            )
                            pts_all_masked = pts_all_masked[sample_indices]
                            images_all_masked = images_all_masked[sample_indices]

                        if pts_gt_all_masked.shape[0] > max_eval_points:
                            sample_indices_gt = np.random.choice(
                                pts_gt_all_masked.shape[0],
                                max_eval_points,
                                replace=False,
                            )
                            pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                    pcd.colors = o3d.utility.Vector3dVector(images_all_masked)
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        np.zeros_like(pts_gt_all_masked)
                    )

                    # ====== voxel grid downsampling ======
                    if args.voxel_size > 0:
                        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=args.voxel_size)
                    # ===========================

                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                        ),
                        pcd,
                    )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    chamfer = chamfer_distance(acc, comp, average=True)
                    chamfer_med = chamfer_distance(acc_med, comp_med, average=True)
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, Chamfer: {chamfer}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, Chamfer_med: {chamfer_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, Chamfer: {chamfer}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, Chamfer_med: {chamfer_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    chamfer_all += chamfer
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    chamfer_all_med += chamfer_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            if name_data.lower() == "scannet":
                if accelerator.is_main_process:
                    input_frame = (
                        args.max_frames
                        if args.max_frames is not None
                        else len(dataset[0])
                    )
                    summarize_scannet_metrics(save_path, input_frame)
                continue

            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    Chamfer:\s*(?P<chamfer>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    Chamfer_med:\s*(?P<chamfer_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args.beta_safe = resolve_beta_safe_default(args)

    main(args)
