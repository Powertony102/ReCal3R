import os
import cv2
import numpy as np
import os.path as osp
from collections import deque
import random
from eval.mv_recon.base import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import eval.mv_recon.dataset_utils.cropping as cropping
from scipy.spatial.transform import Rotation


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def shuffle_deque(dq, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Convert deque to list, shuffle, and convert back
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


def sample_frame_ids(frame_ids, kf_every, max_frames=None, keep_two_frames=True):
    if kf_every < 1:
        raise ValueError(f"kf_every must be >= 1, got {kf_every}")

    sampled = frame_ids[::kf_every]
    if keep_two_frames and len(frame_ids) >= 2 and len(sampled) < 2:
        sampled = [frame_ids[0], frame_ids[-1]]

    if max_frames is not None:
        sampled = sampled[:max_frames]

    return sampled


def _numeric_stem(path):
    stem = osp.splitext(osp.basename(str(path)))[0]
    if stem.isdigit():
        return int(stem)
    try:
        return float(stem)
    except ValueError:
        return stem


def _list_image_paths(images_dir):
    image_paths = []
    if not osp.isdir(images_dir):
        return image_paths
    for name in os.listdir(images_dir):
        if name.lower().endswith(IMG_EXTENSIONS):
            image_paths.append(osp.join(images_dir, name))
    return sorted(image_paths, key=_numeric_stem)


def _load_scannet_pose_dir(pose_dir):
    pose_files = []
    if not osp.isdir(pose_dir):
        return None, None, None
    for name in os.listdir(pose_dir):
        if name.endswith(".txt") and osp.splitext(name)[0].isdigit():
            pose_files.append(osp.join(pose_dir, name))
    pose_files = sorted(pose_files, key=_numeric_stem)

    poses = []
    frame_ids = []
    for pose_file in pose_files:
        try:
            pose = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
        except Exception:
            continue
        if not np.isfinite(pose).all():
            continue
        poses.append(pose)
        frame_ids.append(int(osp.splitext(osp.basename(pose_file))[0]))

    if len(poses) == 0:
        return None, None, None
    poses = np.stack(poses, 0)
    first_pose = poses[0].copy()
    rel_poses = np.linalg.inv(first_pose) @ poses
    return rel_poses.astype(np.float32), first_pose.astype(np.float32), np.array(frame_ids)


def _select_scannet_frames(image_paths, available_pose_frame_ids, max_frames):
    image_frame_ids = [int(osp.splitext(osp.basename(path))[0]) for path in image_paths]
    valid_frame_ids = sorted(set(image_frame_ids) & set(available_pose_frame_ids))
    if max_frames is not None and len(valid_frame_ids) > max_frames:
        if max_frames <= 1:
            valid_frame_ids = valid_frame_ids[:max_frames]
        else:
            first_frame = valid_frame_ids[0]
            remaining_frames = valid_frame_ids[1:]
            step = max(1, len(remaining_frames) // (max_frames - 1))
            valid_frame_ids = [first_frame] + remaining_frames[::step][: max_frames - 1]

    frame_id_to_path = {int(osp.splitext(osp.basename(path))[0]): path for path in image_paths}
    pose_frame_to_idx = {fid: idx for idx, fid in enumerate(available_pose_frame_ids)}
    selected_image_paths = [frame_id_to_path[fid] for fid in valid_frame_ids]
    selected_pose_indices = [pose_frame_to_idx[fid] for fid in valid_frame_ids]
    return valid_frame_ids, selected_image_paths, selected_pose_indices


def _load_scannet_intrinsics(scene_dir, image_shape, prefer_depth=True):
    intrinsic_dir = osp.join(scene_dir, "intrinsic")
    if prefer_depth:
        candidates = [
            osp.join(intrinsic_dir, "intrinsic_depth.txt"),
            osp.join(intrinsic_dir, "intrinsic_color.txt"),
            osp.join(scene_dir, "intrinsic_depth.txt"),
            osp.join(scene_dir, "intrinsic_color.txt"),
        ]
    else:
        candidates = [
            osp.join(intrinsic_dir, "intrinsic_color.txt"),
            osp.join(intrinsic_dir, "intrinsic_depth.txt"),
            osp.join(scene_dir, "intrinsic_color.txt"),
            osp.join(scene_dir, "intrinsic_depth.txt"),
        ]
    for path in candidates:
        if osp.exists(path):
            intrinsics = np.loadtxt(path).astype(np.float32)
            return intrinsics[:3, :3]

    height, width = image_shape[:2]
    focal = 0.8 * max(width, height)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


BONN_INTRINSICS = np.array(
    [
        [542.822841, 0.0, 315.593520],
        [0.0, 542.576870, 237.756098],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _read_bonn_file_list(filename):
    entries = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1:]))
    return entries


def _nearest_bonn_entry(entries, timestamp, max_difference):
    if not entries:
        return None
    timestamps = np.array([entry[0] for entry in entries], dtype=np.float64)
    pos = np.searchsorted(timestamps, timestamp)
    candidates = []
    if pos < len(entries):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best_idx = min(candidates, key=lambda idx: abs(timestamps[idx] - timestamp))
    if abs(timestamps[best_idx] - timestamp) > max_difference:
        return None
    return entries[best_idx]


def _bonn_path(seq_dir, value):
    return value if osp.isabs(value) else osp.join(seq_dir, value)


def _tum_pose_to_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix().astype(np.float32)
    pose[:3, 3] = values[:3].astype(np.float32)
    return pose


def _load_bonn_raw_records(seq_dir, max_time_diff=0.02):
    rgb_entries = _read_bonn_file_list(osp.join(seq_dir, "rgb.txt"))
    depth_entries = _read_bonn_file_list(osp.join(seq_dir, "depth.txt"))
    pose_entries = _read_bonn_file_list(osp.join(seq_dir, "groundtruth.txt"))

    records = []
    for timestamp, rgb_values in rgb_entries:
        depth_entry = _nearest_bonn_entry(depth_entries, timestamp, max_time_diff)
        pose_entry = _nearest_bonn_entry(pose_entries, timestamp, max_time_diff)
        if depth_entry is None or pose_entry is None:
            continue
        rgb_path = _bonn_path(seq_dir, rgb_values[0])
        depth_path = _bonn_path(seq_dir, depth_entry[1][0])
        if not osp.exists(rgb_path) or not osp.exists(depth_path):
            continue
        records.append(
            {
                "timestamp": timestamp,
                "rgb": rgb_path,
                "depth": depth_path,
                "pose": _tum_pose_to_matrix(pose_entry[1]),
                "label": f"{timestamp:.6f}",
            }
        )
    return records


def _load_bonn_prepared_records(seq_dir):
    rgb_dirs = sorted(name for name in os.listdir(seq_dir) if name.startswith("rgb_"))
    for rgb_dir_name in reversed(rgb_dirs):
        suffix = rgb_dir_name.removeprefix("rgb_")
        depth_dir = osp.join(seq_dir, f"depth_{suffix}")
        gt_path = osp.join(seq_dir, f"groundtruth_{suffix}.txt")
        if osp.isdir(depth_dir) and osp.exists(gt_path):
            rgb_dir = osp.join(seq_dir, rgb_dir_name)
            break
    else:
        return []

    rgb_paths = _list_image_paths(rgb_dir)
    depth_paths = _list_image_paths(depth_dir)
    gt = np.loadtxt(gt_path, comments="#", ndmin=2)
    count = min(len(rgb_paths), len(depth_paths), len(gt))

    records = []
    for idx in range(count):
        pose_values = gt[idx, 1:8] if gt.shape[1] >= 8 else gt[idx, :7]
        label = osp.splitext(osp.basename(rgb_paths[idx]))[0]
        records.append(
            {
                "timestamp": float(gt[idx, 0]) if gt.shape[1] >= 8 else float(idx),
                "rgb": rgb_paths[idx],
                "depth": depth_paths[idx],
                "pose": _tum_pose_to_matrix(pose_values),
                "label": label,
            }
        )
    return records


def _load_bonn_records(seq_dir, max_time_diff=0.02):
    raw_files = ["rgb.txt", "depth.txt", "groundtruth.txt"]
    if all(osp.exists(osp.join(seq_dir, filename)) for filename in raw_files):
        return _load_bonn_raw_records(seq_dir, max_time_diff=max_time_diff)
    return _load_bonn_prepared_records(seq_dir)


def _is_bonn_sequence_dir(seq_dir):
    if not osp.isdir(seq_dir) or not osp.basename(seq_dir).startswith("rgbd_bonn_"):
        return False
    raw_files = ["rgb.txt", "depth.txt", "groundtruth.txt"]
    if all(osp.exists(osp.join(seq_dir, filename)) for filename in raw_files):
        return True
    return any(name.startswith("rgb_") for name in os.listdir(seq_dir))


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        max_frames=None,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed
        self.max_frames = max_frames

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = sorted(
            d for d in os.listdir(base_dir) if osp.isdir(osp.join(base_dir, d))
        )

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = sample_frame_ids(
                img_idxs, self.kf_every, max_frames=self.max_frames
            )

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views


class DTU(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=49,
        num_frames=5,
        min_thresh=10,
        max_thresh=30,
        test_id=None,
        full_video=False,
        sample_pairs=False,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sample_pairs = sample_pairs

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        if self.test_id is None:
            self.scene_list = os.listdir(osp.join(base_dir))
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")

        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]

            print(f"Test_id: {self.test_id}")

    def load_cam_mvsnet(self, file, interval_scale=1):
        """read camera txt file"""
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq

        print("Scene ID:", scene_id)

        image_path = osp.join(self.ROOT, scene_id, "images")
        depth_path = osp.join(self.ROOT, scene_id, "depths")
        mask_path = osp.join(self.ROOT, scene_id, "binary_masks")
        cam_path = osp.join(self.ROOT, scene_id, "cams")
        pairs_path = osp.join(self.ROOT, scene_id, "pair.txt")

        if not self.full_video:
            img_idxs = self.sample_pairs(pairs_path, seq_id)
        else:
            img_idxs = sorted(os.listdir(image_path))
            img_idxs = img_idxs[:: self.kf_every]

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace(".jpg", ".npy"))
            campath = osp.join(cam_path, im_idx.replace(".jpg", "_cam.txt"))
            maskpath = osp.join(mask_path, im_idx.replace(".jpg", ".png"))

            rgb_image = imread_cv2(impath)
            depthmap = np.load(depthpath)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, "r"))
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="dtu",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        max_frames=None,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed
        self.max_frames = max_frames

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = sorted(
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        )

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(
                [
                    name
                    for name in os.listdir(os.path.join(self.ROOT, scene_id, "images"))
                    if name.endswith(".png")
                ]
            )
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = sample_frame_ids(
                img_idxs, self.kf_every, max_frames=self.max_frames
            )

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0
            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class Bonn(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        kf_every=1,
        max_frames=None,
        max_time_diff=0.02,
        depth_scale=5000.0,
        intrinsics=None,
        test_id=None,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.kf_every = kf_every
        self.max_frames = max_frames
        self.max_time_diff = max_time_diff
        self.depth_scale = depth_scale
        self.intrinsics = (
            BONN_INTRINSICS.copy()
            if intrinsics is None
            else np.array(
                [
                    [intrinsics[0], 0.0, intrinsics[2]],
                    [0.0, intrinsics[1], intrinsics[3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        self.test_id = test_id
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is not None:
            scene = (
                self.test_id
                if self.test_id.startswith("rgbd_bonn_")
                else f"rgbd_bonn_{self.test_id}"
            )
            self.scene_list = [scene]
        else:
            self.scene_list = sorted(
                seq
                for seq in os.listdir(base_dir)
                if _is_bonn_sequence_dir(osp.join(base_dir, seq))
            )

        print(f"Found {len(self.scene_list)} Bonn sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        seq_dir = osp.join(self.ROOT, scene_id)
        records = _load_bonn_records(seq_dir, max_time_diff=self.max_time_diff)
        if len(records) == 0:
            raise RuntimeError(
                f"No valid Bonn RGB-D frames found for sequence {scene_id}"
            )

        record_indices = sample_frame_ids(
            list(range(len(records))), self.kf_every, max_frames=self.max_frames
        )

        views = []
        for record_idx in record_indices:
            record = records[record_idx]
            impath = record["rgb"]
            depthpath = record["depth"]

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = (
                np.nan_to_num(depthmap.astype(np.float32), 0.0) / self.depth_scale
            )
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            intrinsics_ = self.intrinsics.copy()
            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                crop_bbox = (cx - 112, cy - 112, cx + 112, cy + 112)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=record["pose"].astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="bonn",
                    label=osp.join(scene_id, record["label"]),
                    instance=impath,
                )
            )

        return views


class ScanNet(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        kf_every=1,
        max_frames=None,
        num_scenes=None,
        test_id=None,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.kf_every = kf_every
        self.max_frames = max_frames
        self.num_scenes = num_scenes
        self.test_id = test_id
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is not None:
            self.scene_list = [self.test_id]
        else:
            scenes = sorted(
                d
                for d in os.listdir(base_dir)
                if osp.isdir(osp.join(base_dir, d))
                and osp.isdir(osp.join(base_dir, d, "color"))
                and osp.isdir(osp.join(base_dir, d, "pose"))
            )
            if self.num_scenes is not None and len(scenes) > self.num_scenes:
                sample_interval = max(1, len(scenes) // self.num_scenes)
                scenes = scenes[::sample_interval][: self.num_scenes]
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} ScanNet sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, scene_id)
        image_paths = _list_image_paths(osp.join(scene_dir, "color"))
        poses, first_gt_pose, available_pose_frame_ids = _load_scannet_pose_dir(
            osp.join(scene_dir, "pose")
        )
        if poses is None:
            raise RuntimeError(f"No valid ScanNet poses found for scene {scene_id}")

        if self.kf_every > 1:
            keep_frame_ids = set(available_pose_frame_ids[:: self.kf_every].tolist())
            image_paths = [
                path
                for path in image_paths
                if int(osp.splitext(osp.basename(path))[0]) in keep_frame_ids
            ]

        frame_ids, image_paths, pose_indices = _select_scannet_frames(
            image_paths, available_pose_frame_ids, self.max_frames
        )

        views = []
        for frame_id, impath, pose_idx in zip(frame_ids, image_paths, pose_indices):
            rgb_image = imread_cv2(impath)
            depthpath = osp.join(scene_dir, "depth", f"{frame_id}.png")
            has_depth = osp.exists(depthpath)
            if has_depth:
                depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
                depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
                depthmap[depthmap > 10] = 0
                depthmap[depthmap < 1e-3] = 0
                rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
            else:
                depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            intrinsics_ = _load_scannet_intrinsics(
                scene_dir, depthmap.shape, prefer_depth=has_depth
            )
            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                crop_bbox = (cx - 112, cy - 112, cx + 112, cy + 112)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=np.eye(4, dtype=np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="scannet",
                    label=osp.join(scene_id, str(frame_id)),
                    instance=impath,
                    scannet_frame_id=int(frame_id),
                    scannet_camera_pose=poses[pose_idx].astype(np.float32),
                    scannet_first_gt_pose=first_gt_pose.astype(np.float32),
                )
            )

        return views
