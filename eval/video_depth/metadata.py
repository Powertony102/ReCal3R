import os
import glob
from tqdm import tqdm

SINTEL_ROOT = os.environ.get("EVAL_SINTEL_ROOT", "data/MPI_Sintel/training")

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
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "seq_list_func": lambda img_path: list_bonn_sequences(img_path),
        "save_seq_func": lambda seq: _bonn_suffix(seq),
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
        "img_path": "data/scannetv2",
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
        "img_path": os.path.join(SINTEL_ROOT, "final"),
        "anno_path": os.path.join(SINTEL_ROOT, "camdata_left"),
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
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
}

kitti_numbers = [50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500, 700]
kitti_configs = {
    f"kitti_s1_{num}": {
        "img_path": f"data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_{num}",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    }
    for num in kitti_numbers
}
dataset_metadata.update(kitti_configs)


bonn_numbers = [50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500]
bonn_configs = {
    f"bonn_s1_{num}": {
        "img_path": "data/long_bonn_s1/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"rgb_{num}"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(
            img_path, f"rgbd_bonn_{seq}", f"groundtruth_{num}.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "save_seq_func": lambda seq: _bonn_suffix(seq),
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    }
    for num in bonn_numbers
}
dataset_metadata.update(bonn_configs)


def _normalize_bonn_seq(seq):
    return seq if seq.startswith("rgbd_bonn_") else f"rgbd_bonn_{seq}"


def _bonn_seq_dir(img_path, seq):
    return os.path.join(img_path, _normalize_bonn_seq(seq))


def _bonn_suffix(seq):
    return seq.removeprefix("rgbd_bonn_")


def _timestamp_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(stem)
    except ValueError:
        return stem


def _read_bonn_file_list(filename):
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


def _nearest_bonn_entry(entries, timestamp, max_difference=0.02):
    if not entries:
        return None
    best = min(entries, key=lambda entry: abs(entry[0] - timestamp))
    return best if abs(best[0] - timestamp) <= max_difference else None


def _bonn_path(seq_dir, value):
    return value if os.path.isabs(value) else os.path.join(seq_dir, value)


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


def _bonn_depth_dir_from_rgb_dir(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_dir_name = os.path.basename(dir_path)
    if rgb_dir_name == "rgb":
        return os.path.join(seq_dir, "depth")
    if rgb_dir_name.startswith("rgb_"):
        suffix = rgb_dir_name.removeprefix("rgb_")
        return os.path.join(seq_dir, f"depth_{suffix}")
    return None


def _bonn_depth_path_for_image(image_path, depth_entries=None, max_difference=0.02):
    rgb_dir = os.path.dirname(image_path)
    seq_dir = os.path.dirname(rgb_dir)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    try:
        timestamp = float(stem)
    except ValueError:
        timestamp = None

    if timestamp is not None and depth_entries:
        depth_entry = _nearest_bonn_entry(
            depth_entries, timestamp, max_difference=max_difference
        )
        if depth_entry is not None:
            depth_path = _bonn_path(seq_dir, depth_entry[1][0])
            if os.path.exists(depth_path):
                return depth_path

    depth_dir = _bonn_depth_dir_from_rgb_dir(rgb_dir)
    if depth_dir is None:
        return None
    depth_path = os.path.join(depth_dir, os.path.basename(image_path))
    return depth_path if os.path.exists(depth_path) else None


def bonn_depths_for_images(image_paths, max_difference=0.02):
    if len(image_paths) == 0:
        return []
    seq_dir = os.path.dirname(os.path.dirname(image_paths[0]))
    depth_entries = _read_bonn_file_list(os.path.join(seq_dir, "depth.txt"))
    depth_paths = []
    for image_path in image_paths:
        depth_path = _bonn_depth_path_for_image(
            image_path, depth_entries=depth_entries, max_difference=max_difference
        )
        if depth_path is not None:
            depth_paths.append(depth_path)
    return depth_paths


def list_bonn_images(dir_path):
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(glob.glob(os.path.join(dir_path, ext)))
    images = sorted(images, key=_timestamp_sort_key)

    if len(images) == 0:
        return images

    depth_paths = bonn_depths_for_images(images)
    if len(depth_paths) == len(images):
        return images
    depth_path_set = set(depth_paths)
    return [
        image_path
        for image_path in images
        if _bonn_depth_path_for_image(
            image_path,
            depth_entries=_read_bonn_file_list(
                os.path.join(os.path.dirname(os.path.dirname(image_path)), "depth.txt")
            ),
        )
        in depth_path_set
    ]


def _tum_seq_dir(img_path, seq):
    return os.path.join(img_path, seq)


def _read_tum_file_list(filename):
    return _read_bonn_file_list(filename)


def _nearest_tum_entry(entries, timestamp, max_difference=0.02):
    return _nearest_bonn_entry(entries, timestamp, max_difference=max_difference)


def _tum_path(seq_dir, value):
    return value if os.path.isabs(value) else os.path.join(seq_dir, value)


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


def _tum_depth_dir_from_rgb_dir(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_dir_name = os.path.basename(dir_path)
    if rgb_dir_name == "rgb":
        return os.path.join(seq_dir, "depth")
    if rgb_dir_name.startswith("rgb_"):
        suffix = rgb_dir_name.removeprefix("rgb_")
        return os.path.join(seq_dir, f"depth_{suffix}")
    return None


def tum_depth_path_for_image(image_path, max_difference=0.02):
    rgb_dir = os.path.dirname(image_path)
    seq_dir = os.path.dirname(rgb_dir)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    try:
        timestamp = float(stem)
    except ValueError:
        timestamp = None

    if timestamp is not None:
        depth_entries = _read_tum_file_list(os.path.join(seq_dir, "depth.txt"))
        depth_entry = _nearest_tum_entry(
            depth_entries, timestamp, max_difference=max_difference
        )
        if depth_entry is not None:
            depth_path = _tum_path(seq_dir, depth_entry[1][0])
            if os.path.exists(depth_path):
                return depth_path

    depth_dir = _tum_depth_dir_from_rgb_dir(rgb_dir)
    if depth_dir is None:
        return None
    depth_path = os.path.join(depth_dir, os.path.basename(image_path))
    return depth_path if os.path.exists(depth_path) else None


def tum_depths_for_images(image_paths, max_difference=0.02):
    depth_paths = []
    for image_path in image_paths:
        depth_path = tum_depth_path_for_image(
            image_path, max_difference=max_difference
        )
        if depth_path is not None:
            depth_paths.append(depth_path)
    return depth_paths


def list_tum_images(dir_path):
    seq_dir = os.path.dirname(dir_path)
    rgb_entries = _read_tum_file_list(os.path.join(seq_dir, "rgb.txt"))
    if rgb_entries:
        matched_images = []
        for timestamp, values in rgb_entries:
            image_path = _tum_path(seq_dir, values[0])
            if not os.path.exists(image_path):
                continue
            if tum_depth_path_for_image(image_path) is None:
                continue
            matched_images.append(image_path)
        if matched_images:
            return matched_images

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(glob.glob(os.path.join(dir_path, ext)))
    images = sorted(images, key=_timestamp_sort_key)

    depth_paths = tum_depths_for_images(images)
    if len(depth_paths) == len(images):
        return images
    return [
        image_path for image_path in images if tum_depth_path_for_image(image_path)
    ]


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = list_bonn_images(os.path.join(dir, "rgb"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = list_bonn_sequences(img_path) if args.seq_list is None else args.seq_list
        for seq in tqdm(seq_list):
            filelist = list_bonn_images(bonn_rgb_dir(img_path, seq))
            save_dir = f"{args.output_dir}/{_bonn_suffix(seq)}"
            yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = sorted(glob.glob(f"{img_path}/*"))
    for seq in tqdm(seq_list):
        filelist = sorted(glob.glob(f"{seq}/color_90/*.jpg"))
        save_dir = f"{args.output_dir}/{os.path.basename(seq)}"
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
