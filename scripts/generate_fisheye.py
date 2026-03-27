"""
把現有的棋盤格校正圖加上已知的魚眼畸變，產生模擬魚眼圖片。
用途：在沒有真實魚眼鏡頭的情況下測試 fisheye calibration pipeline。

Ground-truth 參數會存到 data/ground_truth.yaml 供驗證。
"""
import cv2
import numpy as np
import glob
import os
import sys

# Ground-truth fisheye 參數
GT_K = np.array([
    [280.0,   0.0, 320.0],
    [  0.0, 280.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

GT_D = np.array([-0.10, 0.02, -0.005, 0.001], dtype=np.float64).reshape(4, 1)

def apply_fisheye_distortion(img, K, D):
    """對一張圖片加上魚眼畸變"""
    h, w = img.shape[:2]

    # 建立去畸變的反向映射：
    # undistortImage 是從畸變圖 → 無畸變圖
    # 我們要反過來：從無畸變圖 → 畸變圖
    # 方法：用 initUndistortRectifyMap 取得 map，然後反轉

    # 先用 fisheye 模型建立 undistort map
    new_K = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )

    # map1, map2 是 undistort map（畸變圖座標 → 無畸變圖座標）
    # 我們要 distort map（無畸變圖座標 → 畸變圖座標）
    # 方法：對每個像素點做 fisheye projectPoints

    # 建立無畸變圖的網格座標
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    # 轉成歸一化相機座標
    pts_norm = np.zeros((len(pts), 1, 2), dtype=np.float64)
    pts_norm[:, 0, 0] = (pts[:, 0] - K[0, 2]) / K[0, 0]
    pts_norm[:, 0, 1] = (pts[:, 1] - K[1, 2]) / K[1, 1]

    # 用 fisheye.distortPoints 加上畸變
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    # distortPoints: 歸一化座標 → 畸變後的像素座標
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
    dst_dir = os.path.expanduser("~/avm-pipeline/data/calib_images")
    os.makedirs(dst_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not images:
        print(f"錯誤：{src_dir} 裡沒有圖片")
        sys.exit(1)

    print(f"找到 {len(images)} 張原始棋盤格圖")
    print(f"Ground-truth K:\n{GT_K}")
    print(f"Ground-truth D: {GT_D.ravel()}")
    print()

    for i, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            print(f"  跳過：{path}")
            continue

        # 如果是灰階圖，轉成 BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        distorted = apply_fisheye_distortion(img, GT_K, GT_D)

        out_name = f"fisheye_{i:02d}.jpg"
        out_path = os.path.join(dst_dir, out_name)
        cv2.imwrite(out_path, distorted)
        print(f"  ✓ {os.path.basename(path)} → {out_name}")

    # 存 ground truth 參數
    gt_path = os.path.join(os.path.expanduser("~/avm-pipeline/data"),
                           "ground_truth.yaml")
    fs = cv2.FileStorage(gt_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", GT_K)
    fs.write("dist_coeffs", GT_D)
    fs.release()
    print(f"\nGround truth 已儲存至 {gt_path}")
    print(f"畸變圖已輸出至 {dst_dir}/")


if __name__ == "__main__":
    main()
