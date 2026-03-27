"""
產生 4 方向模擬魚眼測試圖 (front/back/left/right)。
用不同顏色 + 方向標記 + 模擬場景元素，然後加上魚眼畸變。
"""
import cv2
import numpy as np
import os
import sys

# 和 generate_fisheye.py 相同的 ground-truth 參數
GT_K = np.array([
    [280.0,   0.0, 320.0],
    [  0.0, 280.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

GT_D = np.array([-0.10, 0.02, -0.005, 0.001], dtype=np.float64).reshape(4, 1)


def apply_fisheye_distortion(img, K, D):
    """和 generate_fisheye.py 相同的畸變函式"""
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
    """繪製模擬場景：地面 + 車道線 + 方向標記"""
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
        # 垂直車道線
        for lx in [w // 3, 2 * w // 3]:
            y = 0
            while y < h:
                cv2.line(img, (lx, y), (lx, min(y + dash_len, h)), line_color, 3)
                y += dash_len + gap_len
    else:
        # 水平車道線
        for ly in [h // 3, 2 * h // 3]:
            x = 0
            while x < w:
                cv2.line(img, (x, ly), (min(x + dash_len, w), ly), line_color, 3)
                x += dash_len + gap_len

    # 畫一些模擬物件（方塊代表車輛/障礙物）
    obj_positions = {
        "front": [(150, 100, 80, 50), (400, 200, 60, 40)],
        "back":  [(200, 300, 70, 45), (450, 150, 55, 35)],
        "left":  [(100, 150, 50, 80), (300, 350, 45, 70)],
        "right": [(500, 100, 50, 80), (250, 300, 45, 70)],
    }

    obj_colors = [(0, 0, 180), (0, 140, 0), (180, 100, 0)]
    for i, (ox, oy, ow, oh) in enumerate(obj_positions.get(direction, [])):
        color = obj_colors[i % len(obj_colors)]
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), color, -1)
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 1)

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

    print("產生 4 方向測試圖（含魚眼畸變）...")

    for name, (bg, text_col) in views.items():
        scene = draw_scene(W, H, name, bg, text_col)
        distorted = apply_fisheye_distortion(scene, GT_K, GT_D)

        out_path = os.path.join(dst_dir, f"{name}.jpg")
        cv2.imwrite(out_path, distorted)
        print(f"  ✓ {name}.jpg")

    print(f"\n測試圖已輸出至 {dst_dir}/")


if __name__ == "__main__":
    main()
