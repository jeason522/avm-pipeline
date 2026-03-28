"""
產生 4 方向模擬魚眼測試圖 (front/back/left/right)。
每顆相機使用各自獨立的 K, D（與 generate_fisheye.py 一致），
並輸出 Ground-truth 外參 (R, t) 到 JSON 供 pipeline 使用。

外參定義：每顆相機相對車體座標系（車體中心在地面）的旋轉和平移。
車體座標系：X 向右，Y 向前，Z 向上。
"""
import cv2
import numpy as np
import os
import json

# ─── 與 generate_fisheye.py 一致的各相機內參 ───
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

# ─── 外參定義 ───
# 車體座標系：X 右, Y 前, Z 上
# 相機座標系：x 右, y 下, z 前（OpenCV 慣例）
# R: 從車體座標到相機座標的旋轉矩陣
# t: 相機在車體座標系中的位置 (meters)

def _Rx(deg):
    r = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

def _Ry(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])

def _Rz(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])

# 車體到相機的轉換：先把車體座標 (X右Y前Z上) 轉成相機座標 (x右y下z前)
# 基礎旋轉：繞 X 軸轉 90° (Y前→z前, Z上→-y下)
R_body_to_cam_base = _Rx(90)  # 基礎：前方相機直視前方

EXTRINSICS = {
    "front": {
        # 前方相機：安裝在車頭中央，俯視 30°
        "R": (_Rx(30) @ R_body_to_cam_base).tolist(),
        "t": [0.0, 2.0, 1.5],  # 車頭中央，離地 1.5m
    },
    "back": {
        # 後方相機：安裝在車尾中央，俯視 30°，面向後方（繞 Z 旋轉 180°）
        "R": (_Rx(30) @ R_body_to_cam_base @ _Rz(180)).tolist(),
        "t": [0.0, -2.0, 1.5],  # 車尾中央
    },
    "left": {
        # 左方相機：安裝在左側後視鏡，俯視 30°，面向左方（繞 Z 旋轉 90°）
        "R": (_Rx(30) @ R_body_to_cam_base @ _Rz(90)).tolist(),
        "t": [-1.0, 1.0, 1.2],  # 左側後視鏡
    },
    "right": {
        # 右方相機：安裝在右側後視鏡，俯視 30°，面向右方（繞 Z 旋轉 -90°）
        "R": (_Rx(30) @ R_body_to_cam_base @ _Rz(-90)).tolist(),
        "t": [1.0, 1.0, 1.2],  # 右側後視鏡
    },
}


def apply_fisheye_distortion(img, K, D):
    """對一張圖片加上魚眼畸變"""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    pts_norm = np.zeros((len(pts), 1, 2), dtype=np.float64)
    pts_norm[:, 0, 0] = (pts[:, 0] - K[0, 2]) / K[0, 0]
    pts_norm[:, 0, 1] = (pts[:, 1] - K[1, 2]) / K[1, 1]

    distorted = cv2.fisheye.distortPoints(pts_norm, K, D)

    dist_map_x = distorted[:, 0, 0].reshape(h, w).astype(np.float32)
    dist_map_y = distorted[:, 0, 1].reshape(h, w).astype(np.float32)

    distorted_img = cv2.remap(img, dist_map_x, dist_map_y,
                               cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    return distorted_img


def draw_scene(w, h, direction, bg_color, text_color):
    """繪製模擬場景：地面 + 車道線 + 方向標記 + 模擬車輛"""
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # 畫地面紋理（模擬柏油路）
    rng = np.random.RandomState(hash(direction) % 2**31)
    noise = rng.randint(0, 20, (h, w), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(int) + noise - 10, 0, 255).astype(np.uint8)

    # 畫車道線（白色虛線）
    line_color = (200, 200, 200)
    dash_len = 40
    gap_len = 30

    if direction in ("front", "back"):
        for lx in [w // 3, 2 * w // 3]:
            y = 0
            while y < h:
                cv2.line(img, (lx, y), (lx, min(y + dash_len, h)), line_color, 3)
                y += dash_len + gap_len
    else:
        for ly in [h // 3, 2 * h // 3]:
            x = 0
            while x < w:
                cv2.line(img, (x, ly), (min(x + dash_len, w), ly), line_color, 3)
                x += dash_len + gap_len

    # 畫模擬車輛（矩形 + 車頂 + 輪子，比彩色方塊更真實）
    vehicle_positions = {
        "front": [(130, 80, 100, 60), (420, 180, 80, 50)],
        "back":  [(180, 280, 90, 55), (430, 130, 70, 45)],
        "left":  [(80, 130, 60, 100), (280, 320, 55, 90)],
        "right": [(480, 80, 60, 100), (230, 280, 55, 90)],
    }

    for i, (ox, oy, ow, oh) in enumerate(vehicle_positions.get(direction, [])):
        # 車體（深色）
        body_color = [(40, 40, 120), (40, 100, 40), (100, 60, 20)][i % 3]
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), body_color, -1)
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (180, 180, 180), 2)
        # 車窗（淺色）
        wx, wy = ox + ow // 6, oy + oh // 6
        ww, wh = ow * 2 // 3, oh // 3
        cv2.rectangle(img, (wx, wy), (wx + ww, wy + wh), (120, 140, 160), -1)
        # 輪子
        wheel_r = min(ow, oh) // 8
        cv2.circle(img, (ox + ow // 5, oy + oh), wheel_r, (30, 30, 30), -1)
        cv2.circle(img, (ox + 4 * ow // 5, oy + oh), wheel_r, (30, 30, 30), -1)

    # 方向標記文字
    label = direction.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 2.0, 3)
    tx = (w - tw) // 2
    ty = h - 30
    cv2.putText(img, label, (tx, ty), font, 2.0, text_color, 3)

    # 箭頭指示方向
    cx, cy = w // 2, h // 2
    arrows = {
        "front": (cx, cy + 40, cx, cy - 40),
        "back":  (cx, cy - 40, cx, cy + 40),
        "left":  (cx + 40, cy, cx - 40, cy),
        "right": (cx - 40, cy, cx + 40, cy),
    }
    ax1, ay1, ax2, ay2 = arrows[direction]
    cv2.arrowedLine(img, (ax1, ay1), (ax2, ay2), (255, 255, 0), 3, tipLength=0.3)

    return img


def main():
    dst_dir = os.path.expanduser("~/avm-pipeline/data/test_views")
    os.makedirs(dst_dir, exist_ok=True)

    W, H = 640, 480

    views = {
        "front": ((80, 80, 80),   (0, 255, 0)),
        "back":  ((70, 70, 70),   (0, 200, 255)),
        "left":  ((75, 75, 75),   (255, 100, 100)),
        "right": ((85, 85, 85),   (100, 100, 255)),
    }

    print("產生 4 方向測試圖（各相機獨立內參 + 魚眼畸變）...")

    for name, (bg, text_col) in views.items():
        K = CAMERA_PARAMS[name]["K"]
        D = CAMERA_PARAMS[name]["D"]

        scene = draw_scene(W, H, name, bg, text_col)
        distorted = apply_fisheye_distortion(scene, K, D)

        out_path = os.path.join(dst_dir, f"{name}.jpg")
        cv2.imwrite(out_path, distorted)
        print(f"  ✓ {name}.jpg (fx={K[0,0]:.0f})")

    # 儲存 GT 外參到 JSON
    extrinsics_path = os.path.join(
        os.path.expanduser("~/avm-pipeline/data"), "extrinsics_gt.json"
    )
    with open(extrinsics_path, "w") as f:
        json.dump(EXTRINSICS, f, indent=2)
    print(f"\nGT 外參已儲存至 {extrinsics_path}")

    # 儲存各相機內參 GT 到 JSON（供驗證用）
    intrinsics_gt = {}
    for name, params in CAMERA_PARAMS.items():
        intrinsics_gt[name] = {
            "K": params["K"].tolist(),
            "D": params["D"].ravel().tolist(),
        }
    intrinsics_path = os.path.join(
        os.path.expanduser("~/avm-pipeline/data"), "intrinsics_gt.json"
    )
    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics_gt, f, indent=2)
    print(f"GT 內參已儲存至 {intrinsics_path}")

    print(f"\n測試圖已輸出至 {dst_dir}/")


if __name__ == "__main__":
    main()
