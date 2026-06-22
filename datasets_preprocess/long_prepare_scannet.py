import glob
import os
import shutil
import numpy as np

# configurable parameters
for TARGET_FRAMES in [50,90,100,150,200,300,400,500,600,700,800,900,1000]:

    SAMPLE_INTERVAL = 3  # sampling interval, take 1 frame every N frames original 3

    seq_list = sorted(os.listdir("/home/share/Dataset/3D_scene/ScanNet/scans_test/"))

    for seq in seq_list:
        img_pathes = sorted(glob.glob(f"/home/share/Dataset/3D_scene/ScanNet/scans_test/{seq}/color/*.jpg"), key=lambda x: int(os.path.basename(x).split('.')[0]))
        depth_pathes = sorted(glob.glob(f"/home/share/Dataset/3D_scene/ScanNet/scans_test/{seq}/depth/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
        pose_pathes = sorted(glob.glob(f"/home/share/Dataset/3D_scene/ScanNet/scans_test/{seq}/pose/*.txt"), key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # calculate the required original frame count
        required_frames = TARGET_FRAMES * SAMPLE_INTERVAL
        total_frames = min(len(img_pathes), len(depth_pathes), len(pose_pathes))
        
        # if the original frame count is not enough, adjust the target frame count
        actual_target_frames = min(TARGET_FRAMES, total_frames // SAMPLE_INTERVAL)
        
        print(f"{seq}: original frame count {total_frames}, target frames {TARGET_FRAMES}, actual frames {actual_target_frames}")

        # use target frame count to name the directory
        new_color_dir = f"./data/long_scannet_s{SAMPLE_INTERVAL}/{seq}/color_{TARGET_FRAMES}"
        new_depth_dir = f"./data/long_scannet_s{SAMPLE_INTERVAL}/{seq}/depth_{TARGET_FRAMES}"

        # sample according to the target frame count
        new_img_pathes = img_pathes[:actual_target_frames*SAMPLE_INTERVAL:SAMPLE_INTERVAL]
        new_depth_pathes = depth_pathes[:actual_target_frames*SAMPLE_INTERVAL:SAMPLE_INTERVAL]
        new_pose_pathes = pose_pathes[:actual_target_frames*SAMPLE_INTERVAL:SAMPLE_INTERVAL]

        # if the target directory exists, delete it
        if os.path.exists(new_color_dir):
            shutil.rmtree(new_color_dir)
        if os.path.exists(new_depth_dir):
            shutil.rmtree(new_depth_dir)
        
        os.makedirs(new_color_dir, exist_ok=True)
        os.makedirs(new_depth_dir, exist_ok=True)

        for i, (img_path, depth_path) in enumerate(zip(new_img_pathes, new_depth_pathes)):
            shutil.copy(img_path, f"{new_color_dir}/frame_{i:04d}.jpg")
            shutil.copy(depth_path, f"{new_depth_dir}/frame_{i:04d}.png")

        # use target frame count to name the pose file
        pose_new_path = f"./data/long_scannet_s{SAMPLE_INTERVAL}/{seq}/pose_{TARGET_FRAMES}.txt"
        with open(pose_new_path, 'w') as f:
            for i, pose_path in enumerate(new_pose_pathes):
                with open(pose_path, 'r') as pose_file:
                    pose = np.loadtxt(pose_file)
                    pose = pose.reshape(-1)
                    f.write(f"{' '.join(map(str, pose))}\n")
