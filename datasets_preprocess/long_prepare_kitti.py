import shutil
import os
from pathlib import Path


DATASET_ROOT = Path(os.environ.get("KITTI_ROOT", "data/kitti"))
OUTPUT_BASE = Path("./data/long_kitti_s1/depth_selection/val_selection_cropped")
TARGET_FRAMES_LIST = [100, 300, 500, 700]
CAMERA_NAME = "image_02"


def depth_dir_list(dataset_root):
    return sorted(
        dataset_root.glob(f"val/*/proj_depth/groundtruth/{CAMERA_NAME}")
    )


def extract_drive_name(depth_dir):
    # .../val/<drive_name>/proj_depth/groundtruth/image_02
    return depth_dir.parents[2].name


def extract_date_name(drive_name):
    return "_".join(drive_name.split("_")[:3])


def resolve_rgb_path(dataset_root, drive_name, depth_filename):
    date_name = extract_date_name(drive_name)
    return (
        dataset_root
        / date_name
        / drive_name
        / CAMERA_NAME
        / "data"
        / depth_filename
    )


def remove_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prune_raw_dataset(dataset_root):
    """
    Keep only raw files needed by this KITTI long subset pipeline:
    - val/<drive>/proj_depth/groundtruth/image_02/*.png
    - <date>/<drive>/image_02/data/*.png for drives in val split
    """
    val_root = dataset_root / "val"
    if not val_root.exists():
        raise FileNotFoundError(f"Missing val split under dataset root: {val_root}")

    val_drives = sorted(
        drive_dir.name for drive_dir in val_root.iterdir() if drive_dir.is_dir()
    )
    keep_dates = {extract_date_name(drive_name) for drive_name in val_drives}

    print(f"[PRUNE] Val drives: {len(val_drives)}, keep dates: {sorted(keep_dates)}")

    # Top-level cleanup: keep val + needed date folders + tiny txt descriptors.
    for child in dataset_root.iterdir():
        if child.name == "val":
            continue
        if child.is_dir() and child.name in keep_dates:
            continue
        if child.is_file() and child.suffix.lower() == ".txt":
            continue
        print(f"[PRUNE] Remove top-level unused entry: {child}")
        remove_path(child)

    # Keep only val drives and depth GT image_02 in val split.
    for drive_dir in val_root.iterdir():
        if not drive_dir.is_dir():
            if drive_dir.suffix.lower() != ".txt":
                print(f"[PRUNE] Remove val-side file: {drive_dir}")
                remove_path(drive_dir)
            continue
        if drive_dir.name not in val_drives:
            print(f"[PRUNE] Remove unexpected val drive: {drive_dir}")
            remove_path(drive_dir)
            continue

        keep = drive_dir / "proj_depth" / "groundtruth" / CAMERA_NAME
        for child in drive_dir.iterdir():
            if child.name != "proj_depth":
                print(f"[PRUNE] Remove val drive child: {child}")
                remove_path(child)
        proj_depth_dir = drive_dir / "proj_depth"
        if proj_depth_dir.exists():
            for child in proj_depth_dir.iterdir():
                if child.name != "groundtruth":
                    print(f"[PRUNE] Remove proj_depth child: {child}")
                    remove_path(child)
        groundtruth_dir = proj_depth_dir / "groundtruth"
        if groundtruth_dir.exists():
            for child in groundtruth_dir.iterdir():
                if child.name != CAMERA_NAME:
                    print(f"[PRUNE] Remove groundtruth child: {child}")
                    remove_path(child)
        if keep.exists():
            for file_path in keep.iterdir():
                if file_path.suffix.lower() != ".png":
                    print(f"[PRUNE] Remove non-png depth file: {file_path}")
                    remove_path(file_path)

    # Keep only val drives and image_02/data in raw date folders.
    for date_name in sorted(keep_dates):
        date_dir = dataset_root / date_name
        if not date_dir.exists():
            continue
        for child in date_dir.iterdir():
            if child.is_file():
                if child.suffix.lower() != ".txt":
                    print(f"[PRUNE] Remove date-side file: {child}")
                    remove_path(child)
                continue

            if child.name not in val_drives:
                print(f"[PRUNE] Remove non-val raw drive: {child}")
                remove_path(child)
                continue

            drive_dir = child
            for drive_child in drive_dir.iterdir():
                if drive_child.name != CAMERA_NAME:
                    print(f"[PRUNE] Remove raw drive child: {drive_child}")
                    remove_path(drive_child)
            image_dir = drive_dir / CAMERA_NAME
            if image_dir.exists():
                for image_child in image_dir.iterdir():
                    if image_child.name != "data":
                        print(f"[PRUNE] Remove image_02 child: {image_child}")
                        remove_path(image_child)
            data_dir = image_dir / "data"
            if data_dir.exists():
                for file_path in data_dir.iterdir():
                    if file_path.suffix.lower() != ".png":
                        print(f"[PRUNE] Remove non-png RGB file: {file_path}")
                        remove_path(file_path)


def cleanup_unused_outputs(output_base, keep_targets):
    output_base.mkdir(parents=True, exist_ok=True)

    for prefix in ("groundtruth_depth_gathered_", "image_gathered_"):
        for path in output_base.glob(f"{prefix}*"):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix(prefix)
            if not suffix.isdigit():
                continue
            if int(suffix) not in keep_targets:
                print(f"[CLEAN] Remove unused prepared dir: {path}")
                shutil.rmtree(path)


def recreate_target_dirs(output_base, target_frames):
    gt_root = output_base / f"groundtruth_depth_gathered_{target_frames}"
    rgb_root = output_base / f"image_gathered_{target_frames}"

    for root in (gt_root, rgb_root):
        if root.exists():
            print(f"[CLEAN] Rebuild target dir: {root}")
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    return gt_root, rgb_root


def collect_valid_pairs(dataset_root, depth_dir):
    drive_name = extract_drive_name(depth_dir)
    all_depth_files = sorted(depth_dir.glob("*.png"))
    pairs = []

    for depth_file in all_depth_files:
        rgb_file = resolve_rgb_path(dataset_root, drive_name, depth_file.name)
        if rgb_file.exists():
            pairs.append((depth_file, rgb_file))
        else:
            print(f"[WARN] Missing RGB for {depth_file}: {rgb_file}")

    return drive_name, pairs


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    prune_raw_dataset(DATASET_ROOT)

    cleanup_unused_outputs(OUTPUT_BASE, set(TARGET_FRAMES_LIST))

    depth_dirs = depth_dir_list(DATASET_ROOT)
    if len(depth_dirs) == 0:
        raise RuntimeError(
            f"No depth directories found under {DATASET_ROOT}/val/*/proj_depth/groundtruth/{CAMERA_NAME}"
        )

    drive_to_pairs = {}
    for depth_dir in depth_dirs:
        drive_name, pairs = collect_valid_pairs(DATASET_ROOT, depth_dir)
        drive_to_pairs[drive_name] = pairs
        print(
            f"[SCAN] {drive_name}: valid RGB-depth pairs {len(pairs)} "
            f"(from {len(list(depth_dir.glob('*.png')))} depth files)"
        )

    for target_frames in TARGET_FRAMES_LIST:
        gt_root, rgb_root = recreate_target_dirs(OUTPUT_BASE, target_frames)

        for drive_name, pairs in drive_to_pairs.items():
            selected_pairs = pairs[:target_frames]
            if len(selected_pairs) == 0:
                print(f"[SKIP] {drive_name}: no valid pairs for target {target_frames}")
                continue

            seq_name = f"{drive_name}_02"
            gt_seq_dir = gt_root / seq_name
            rgb_seq_dir = rgb_root / seq_name
            gt_seq_dir.mkdir(parents=True, exist_ok=True)
            rgb_seq_dir.mkdir(parents=True, exist_ok=True)

            for depth_file, rgb_file in selected_pairs:
                shutil.copy2(depth_file, gt_seq_dir / depth_file.name)
                shutil.copy2(rgb_file, rgb_seq_dir / rgb_file.name)

            print(
                f"[DONE] {seq_name}: target {target_frames}, copied {len(selected_pairs)} pairs"
            )

    print(
        "[FINISH] Prepared KITTI long subsets. "
        "Raw dataset was pruned to files used by this pipeline before generation."
    )


if __name__ == "__main__":
    main()
