import os
import glob
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "kitti": {
        "img_path": "data/kitti/depth_selection/val_selection_cropped/image_gathered",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: bonn_rgb_dir(img_path, seq),
        "filelist_func": lambda dir_path: list_bonn_images(dir_path),
        "gt_traj_func": lambda img_path, anno_path, seq: bonn_gt_path(img_path, seq),
        "traj_format": "bonn",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_bonn_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": os.environ.get(
            "EVAL_SCANNET_ROOT",
            "data/scannetv2/process_scannet",
        ),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: scannet_color_dir(img_path, seq),
        "filelist_func": lambda dir_path: list_scannet_images(dir_path),
        "gt_traj_func": lambda img_path, anno_path, seq: scannet_pose_path(
            img_path, seq
        ),
        "traj_format": "scannet",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_scannet_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-257": {
        "img_path": "data/scannetv2_3_257",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-129": {
        "img_path": "data/scannetv2_3_129",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-65": {
        "img_path": "data/scannetv2_3_65",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-33": {
        "img_path": "data/scannetv2_3_33",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: tum_rgb_dir(img_path, seq),
        "filelist_func": lambda dir_path: list_tum_images(dir_path),
        "gt_traj_func": lambda img_path, anno_path, seq: tum_gt_path(img_path, seq),
        "traj_format": "tum",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_tum_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
    "7scenes": {
        "img_path": os.environ.get("EVAL_7SCENES_ROOT", "data/7scenes"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "filelist_func": lambda dir_path: list_seven_scenes_images(dir_path),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq),
        "traj_format": "7scenes",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_seven_scenes_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "nrgbd": {
        "img_path": os.environ.get("EVAL_NRGBD_ROOT", "data/nrgbd"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "images"),
        "filelist_func": lambda dir_path: list_nrgbd_images(dir_path),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "poses.txt"
        ),
        "traj_format": "nrgbd",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_nrgbd_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
}



scannet_numbers = [50, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
scannet_configs = {
    f"scannet_s3_{num}": {
        "img_path": "data/long_scannet_s3",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"color_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, seq, f"pose_{num}.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    }
    for num in scannet_numbers
}
# then update dataset_metadata
dataset_metadata.update(scannet_configs)

tum_numbers = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
tum_configs = {
    f"tum_s1_{num}": {
        "img_path": "data/long_tum_s1",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"rgb_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, seq, f"groundtruth_{num}.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_tum_sequences(img_path),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "filelist_func": lambda dir_path: list_tum_images(dir_path),
        "process_func": None,
        }
    for num in tum_numbers
}
dataset_metadata.update(tum_configs)


def _normalize_bonn_seq(seq):
    return seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}"


def _bonn_seq_dir(img_path, seq):
    return os.path.join(img_path, _normalize_bonn_seq(seq))


def _bonn_suffix(seq):
    return seq.removeprefix("rgbd_bonn_")


def list_bonn_sequences(img_path):
    seqs = []
    if not os.path.isdir(img_path):
        return seqs
    for seq in sorted(os.listdir(img_path)):
        seq_dir = os.path.join(img_path, seq)
        if not os.path.isdir(seq_dir) or not seq.startswith("rgbd_bonn_"):
            continue
        if os.path.isdir(os.path.join(seq_dir, "rgb")) or glob.glob(
            os.path.join(seq_dir, "rgb_*")
        ):
            seqs.append(_bonn_suffix(seq))
    return seqs


def bonn_rgb_dir(img_path, seq):
    seq_dir = _bonn_seq_dir(img_path, seq)
    raw_rgb_dir = os.path.join(seq_dir, "rgb")
    if os.path.isdir(raw_rgb_dir):
        return raw_rgb_dir
    prepared_rgb_dir = os.path.join(seq_dir, "rgb_110")
    if os.path.isdir(prepared_rgb_dir):
        return prepared_rgb_dir
    candidates = sorted(glob.glob(os.path.join(seq_dir, "rgb_*")))
    if candidates:
        return candidates[-1]
    return raw_rgb_dir


def bonn_gt_path(img_path, seq):
    seq_dir = _bonn_seq_dir(img_path, seq)
    raw_gt_path = os.path.join(seq_dir, "groundtruth.txt")
    if os.path.exists(raw_gt_path):
        return raw_gt_path
    prepared_gt_path = os.path.join(seq_dir, "groundtruth_110.txt")
    if os.path.exists(prepared_gt_path):
        return prepared_gt_path
    candidates = sorted(glob.glob(os.path.join(seq_dir, "groundtruth_*.txt")))
    if candidates:
        return candidates[-1]
    return raw_gt_path


def _bonn_gt_path_from_rgb_dir(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_dir_name = os.path.basename(dir_path)
    if rgb_dir_name == "rgb":
        return os.path.join(seq_dir, "groundtruth.txt")
    if rgb_dir_name.startswith("rgb_"):
        suffix = rgb_dir_name.removeprefix("rgb_")
        return os.path.join(seq_dir, f"groundtruth_{suffix}.txt")
    return None


def _read_bonn_gt_timestamps(gt_path):
    timestamps = []
    if gt_path is None or not os.path.exists(gt_path):
        return timestamps
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 8:
                continue
            try:
                timestamps.append(float(parts[0]))
            except ValueError:
                continue
    return timestamps


def _has_bonn_gt_match(timestamp, gt_timestamps, max_difference=0.02):
    if len(gt_timestamps) == 0:
        return False
    best = min(gt_timestamps, key=lambda gt_timestamp: abs(gt_timestamp - timestamp))
    return abs(best - timestamp) <= max_difference


def list_bonn_images(dir_path):
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return float(stem)
        except ValueError:
            return stem

    images = sorted(images, key=sort_key)
    gt_timestamps = _read_bonn_gt_timestamps(_bonn_gt_path_from_rgb_dir(dir_path))
    if len(gt_timestamps) == 0:
        return images

    matched_images = []
    for path in images:
        try:
            timestamp = float(os.path.splitext(os.path.basename(path))[0])
        except ValueError:
            return images
        if _has_bonn_gt_match(timestamp, gt_timestamps):
            matched_images.append(path)

    return matched_images if matched_images else images


def _tum_seq_dir(img_path, seq):
    return os.path.join(img_path, seq)


def _timestamp_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(stem)
    except ValueError:
        return stem


def _read_tum_file_list(filename):
    entries = []
    if not os.path.exists(filename):
        return entries
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                entries.append((float(parts[0]), parts[1:]))
            except ValueError:
                continue
    return entries


def _nearest_tum_entry(entries, timestamp, max_difference=0.02):
    if not entries:
        return None
    best = min(entries, key=lambda entry: abs(entry[0] - timestamp))
    return best if abs(best[0] - timestamp) <= max_difference else None


def _tum_path(seq_dir, value):
    return value if os.path.isabs(value) else os.path.join(seq_dir, value)


def _tum_gt_path_from_rgb_dir(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_dir_name = os.path.basename(dir_path)
    if rgb_dir_name == "rgb":
        return os.path.join(seq_dir, "groundtruth.txt")
    if rgb_dir_name.startswith("rgb_"):
        suffix = rgb_dir_name.removeprefix("rgb_")
        return os.path.join(seq_dir, f"groundtruth_{suffix}.txt")
    return os.path.join(seq_dir, "groundtruth.txt")


def _read_tum_gt_timestamps(gt_path):
    return [timestamp for timestamp, _ in _read_tum_file_list(gt_path)]


def _has_tum_gt_match(timestamp, gt_timestamps, max_difference=0.02):
    if len(gt_timestamps) == 0:
        return False
    best = min(gt_timestamps, key=lambda gt_timestamp: abs(gt_timestamp - timestamp))
    return abs(best - timestamp) <= max_difference


def list_tum_sequences(img_path):
    seqs = []
    if not os.path.isdir(img_path):
        return seqs
    for seq in sorted(os.listdir(img_path)):
        seq_dir = _tum_seq_dir(img_path, seq)
        if not os.path.isdir(seq_dir):
            continue
        has_raw_layout = os.path.isdir(os.path.join(seq_dir, "rgb")) and os.path.exists(
            os.path.join(seq_dir, "rgb.txt")
        )
        has_prepared_layout = len(glob.glob(os.path.join(seq_dir, "rgb_*"))) > 0
        if has_raw_layout or has_prepared_layout:
            seqs.append(seq)
    return seqs


def tum_rgb_dir(img_path, seq):
    seq_dir = _tum_seq_dir(img_path, seq)
    raw_rgb_dir = os.path.join(seq_dir, "rgb")
    if os.path.isdir(raw_rgb_dir):
        return raw_rgb_dir
    prepared_rgb_dir = os.path.join(seq_dir, "rgb_90")
    if os.path.isdir(prepared_rgb_dir):
        return prepared_rgb_dir
    candidates = sorted(glob.glob(os.path.join(seq_dir, "rgb_*")))
    if candidates:
        return candidates[-1]
    return raw_rgb_dir


def tum_gt_path(img_path, seq):
    seq_dir = _tum_seq_dir(img_path, seq)
    raw_gt_path = os.path.join(seq_dir, "groundtruth.txt")
    if os.path.exists(raw_gt_path):
        return raw_gt_path
    prepared_gt_path = os.path.join(seq_dir, "groundtruth_90.txt")
    if os.path.exists(prepared_gt_path):
        return prepared_gt_path
    candidates = sorted(glob.glob(os.path.join(seq_dir, "groundtruth_*.txt")))
    if candidates:
        return candidates[-1]
    return raw_gt_path


def list_tum_images(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_txt = os.path.join(seq_dir, "rgb.txt")
    rgb_entries = _read_tum_file_list(rgb_txt)
    if rgb_entries:
        gt_timestamps = _read_tum_gt_timestamps(os.path.join(seq_dir, "groundtruth.txt"))
        matched_images = []
        for timestamp, values in rgb_entries:
            if gt_timestamps and not _has_tum_gt_match(timestamp, gt_timestamps):
                continue
            image_path = _tum_path(seq_dir, values[0])
            if os.path.exists(image_path):
                matched_images.append(image_path)
        if matched_images:
            return matched_images

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(glob.glob(os.path.join(dir_path, ext)))
    images = sorted(images, key=_timestamp_sort_key)

    gt_timestamps = _read_tum_gt_timestamps(_tum_gt_path_from_rgb_dir(dir_path))
    if len(gt_timestamps) == 0:
        return images

    matched_images = []
    for path in images:
        try:
            timestamp = float(os.path.splitext(os.path.basename(path))[0])
        except ValueError:
            return images
        if _has_tum_gt_match(timestamp, gt_timestamps):
            matched_images.append(path)

    return matched_images if matched_images else images


def list_seven_scenes_sequences(img_path):
    seqs = []
    for scene in sorted(os.listdir(img_path)):
        scene_dir = os.path.join(img_path, scene)
        split_file = os.path.join(scene_dir, "TestSplit.txt")
        if not os.path.isdir(scene_dir) or not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            split_seqs = f.read().splitlines()
        for seq_id in split_seqs:
            num_part = "".join(filter(str.isdigit, seq_id))
            seq_id = f"seq-{num_part.zfill(2)}"
            seqs.append(f"{scene}/{seq_id}")
    return seqs


def list_nrgbd_sequences(img_path):
    return sorted(
        d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))
    )


def list_seven_scenes_images(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "frame-*.color.png")))


def list_nrgbd_images(dir_path):
    def frame_id(path):
        name = os.path.basename(path)
        return int(name.removeprefix("img").removesuffix(".png"))

    return sorted(glob.glob(os.path.join(dir_path, "img*.png")), key=frame_id)


def _numeric_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else stem


def _valid_scannet_pose_ids(pose_dir):
    valid_ids = []
    if not os.path.isdir(pose_dir):
        return valid_ids
    for pose_path in sorted(glob.glob(os.path.join(pose_dir, "*.txt")), key=_numeric_stem):
        stem = os.path.splitext(os.path.basename(pose_path))[0]
        if not stem.isdigit():
            continue
        try:
            import numpy as np

            pose = np.loadtxt(pose_path).reshape(4, 4)
        except Exception:
            continue
        if np.isfinite(pose).all():
            valid_ids.append(int(stem))
    return valid_ids


def list_scannet_sequences(img_path):
    seqs = []
    for seq in sorted(os.listdir(img_path)):
        seq_dir = os.path.join(img_path, seq)
        if not os.path.isdir(seq_dir):
            continue
        if os.path.isdir(os.path.join(seq_dir, "color")) and os.path.isdir(
            os.path.join(seq_dir, "pose")
        ):
            seqs.append(seq)
        elif os.path.isdir(os.path.join(seq_dir, "color_90")) and os.path.exists(
            os.path.join(seq_dir, "pose_90.txt")
        ):
            seqs.append(seq)
    return seqs


def scannet_color_dir(img_path, seq):
    scene_dir = os.path.join(img_path, seq)
    processed_dir = os.path.join(scene_dir, "color")
    if os.path.isdir(processed_dir):
        return processed_dir
    return os.path.join(scene_dir, "color_90")


def scannet_pose_path(img_path, seq):
    scene_dir = os.path.join(img_path, seq)
    processed_pose_dir = os.path.join(scene_dir, "pose")
    if os.path.isdir(processed_pose_dir):
        return processed_pose_dir
    return os.path.join(scene_dir, "pose_90.txt")


def list_scannet_images(dir_path):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
    image_paths = sorted(image_paths, key=_numeric_stem)

    if os.path.basename(dir_path) == "color":
        pose_dir = os.path.join(os.path.dirname(dir_path), "pose")
        valid_pose_ids = set(_valid_scannet_pose_ids(pose_dir))
        if valid_pose_ids:
            image_paths = [
                path
                for path in image_paths
                if os.path.splitext(os.path.basename(path))[0].isdigit()
                and int(os.path.splitext(os.path.basename(path))[0]) in valid_pose_ids
            ]
    return image_paths


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/rgb/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = (
            ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
            if args.seq_list is None
            else args.seq_list
        )
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_110/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = list_scannet_sequences(img_path)
    for seq in tqdm(seq_list):
        filelist = list_scannet_images(scannet_color_dir(img_path, seq))
        save_dir = f"{args.output_dir}/{seq}"
        yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ]
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir
