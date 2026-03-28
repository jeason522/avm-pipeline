"""
為 4 顆相機分別產生具有不同魚眼畸變的棋盤格校正圖。
每顆相機使用獨立的 Ground-truth K, D，模擬真實系統中
前/後/左/右鏡頭各有不同內參的情況。

Ground-truth 參數會存到 data/calib/<cam>/ground_truth.yaml 供驗證。
"""
import cv2
import numpy as np
import glob
import os
import sys
import json

# 4 顆相機各自的 Ground-truth 魚眼參數
# 模擬不同焦距、不同光學中心、不同畸變係數
CAMERA_PARAMS = {
    "front": {
        "K": np.array([
            [280.0,   0.0, 320.0],
            [  0.0, 280.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64),
        "D": np.array([-0.08, 0.015, -0.003, 0.0008], dtype=np.float64).reshape(4, 1),
    },
    "back": {
        "K": np.array([
            [270.0,   0.0, 325.0],
            [  0.0, 270.0, 238.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64),
        "D": np.array([-0.085, 0.016, -0.004, 0.0009], dtype=np.float64).reshape(4, 1),
    },
    "left": {
        "K": np.array([
            [275.0,   0.0, 318.0],
            [  0.0, 275.0, 242.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64),
        "D": np.array([-0.082, 0.014, -0.003, 0.0007], dtype=np.float64).reshape(4, 1),
    },
    "right": {
        "K": np.array([
            [285.0,   0.0, 322.0],
            [  0.0, 285.0, 236.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64),
        "D": np.array([-0.075, 0.013, -0.003, 0.0006], dtype=np.float64).reshape(4, 1),
    },
}


def apply_fisheye_distortion(img, K, D):
    """對一張圖片加上魚眼畸變"""
    h, w = img.shape[:2]

    # 建立無畸變圖的網格座標
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    # 轉成歸一化相機座標
    pts_norm = np.zeros((len(pts), 1, 2), dtype=np.float64)
    pts_norm[:, 0, 0] = (pts[:, 0] - K[0, 2]) / K[0, 0]
    pts_norm[:, 0, 1] = (pts[:, 1] - K[1, 2]) / K[1, 1]

    # 用 fisheye.distortPoints 加上畸變
    distorted = cv2.fisheye.distortPoints(pts_norm, K, D)

    # 建立 remap 用的 map
    dist_map_x = distorted[:, 0, 0].reshape(h, w).astype(np.float32)
    dist_map_y = distorted[:, 0, 1].reshape(h, w).astype(np.float32)

    # 用 remap 產生畸變圖
    distorted_img = cv2.remap(img, dist_map_x, dist_map_y,
                               cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    return distorted_img


def main():
    src_dir = os.path.expanduser("~/camera_calibration/images")
    base_dir = os.path.expanduser("~/avm-pipeline/data/calib")

    images = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not images:
        print(f"錯誤：{src_dir} 裡沒有圖片")
        sys.exit(1)

    print(f"找到 {len(images)} 張原始棋盤格圖")
    print(f"將為 4 顆相機分別產生畸變校正圖\n")

    for cam_name, params in CAMERA_PARAMS.items():
        K = params["K"]
        D = params["D"]
        dst_dir = os.path.join(base_dir, cam_name)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"── {cam_name} 相機 ──")
        print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        print(f"  D: {D.ravel()}")

        for i, path in enumerate(images):
            img = cv2.imread(path)
            if img is None:
                continue

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            distorted = apply_fisheye_distortion(img, K, D)

            out_name = f"fisheye_{i:02d}.jpg"
            out_path = os.path.join(dst_dir, out_name)
            cv2.imwrite(out_path, distorted)

        # 存 ground truth 參數
        gt_path = os.path.join(dst_dir, "ground_truth.yaml")
        fs = cv2.FileStorage(gt_path, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("dist_coeffs", D)
        fs.release()

        print(f"  ✓ {len(images)} 張畸變圖 → {dst_dir}/")
        print(f"  ✓ GT 參數 → {gt_path}\n")

    # 同時保留舊的 calib_images 目錄（向後相容）
    compat_dir = os.path.expanduser("~/avm-pipeline/data/calib_images")
    if not os.path.exists(compat_dir):
        os.makedirs(compat_dir, exist_ok=True)
        front_params = CAMERA_PARAMS["front"]
        for i, path in enumerate(images):
            img = cv2.imread(path)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            distorted = apply_fisheye_distortion(img, front_params["K"], front_params["D"])
            cv2.imwrite(os.path.join(compat_dir, f"fisheye_{i:02d}.jpg"), distorted)

    print("完成！4 顆相機的校正圖已產生。")


if __name__ == "__main__":
    main()
