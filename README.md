# AVM Pipeline — Fisheye Calibration → Bird's-Eye Stitching → Object Detection

A complete C++ pipeline for automotive Around View Monitor (AVM) systems:
per-camera fisheye calibration (intrinsic + extrinsic), undistortion,
calibration-derived bird's-eye view stitching with feather blending,
and YOLOv8 object detection — all in one executable.

## Pipeline

```
4 Fisheye Camera Images (front / back / left / right)
          │
          ▼
  ┌──────────────────────────────┐
  │  Step 1: Intrinsic Calib     │  cv::fisheye::calibrate()
  │  → Per-camera K + D          │  4 independent calibrations
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Step 1.5: Extrinsic Load    │  JSON config (R, t per camera)
  │  → Camera-to-vehicle pose    │  rotation + translation
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Step 2: Undistort           │  cv::fisheye::undistortImage()
  │  → 4 rectified views         │  per-camera K, D
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Step 3: Bird's-Eye          │  H = K·[r₁ r₂ -Rt]·H_bev
  │  Stitching + Blending        │  calibration-derived homography
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Step 4: YOLO Detect         │  YOLOv8n via ONNX Runtime (C++)
  │  → bounding boxes            │  NMS + 80-class COCO detection
  └──────────┬───────────────────┘
             ▼
       output/result.jpg
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Per-camera intrinsic calibration | `cv::fisheye::calibrate()` with per-image RPE analysis, 4 independent K+D |
| Extrinsic calibration | Camera-to-vehicle R, t loaded from JSON config |
| Undistortion | `cv::fisheye::estimateNewCameraMatrixForUndistortRectify()`, per-camera |
| Calibration-derived IPM | Homography computed from K, R, t and ground plane (Z=0) |
| Feather blending | Distance-weighted alpha masks for seamless stitching |
| Luminance compensation | Gain adjustment to match brightness across views |
| Corner filling | Blended interpolation from adjacent views |
| Object detection | YOLOv8n ONNX, hand-parsed (1,84,8400) tensor + NMS |
| Fallback mode | `--skip-calib` uses raw images with hardcoded trapezoid IPM |

## Build & Run

```bash
# 1. Install dependencies
sudo apt install build-essential cmake libopencv-dev

# 2. Install ONNX Runtime (if not already)
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# 3. Export YOLOv8 model (one-time, requires Python)
pip install ultralytics
python3 scripts/export_onnx.py
mv yolov8n.onnx data/

# 4. Build
git clone https://github.com/jeason522/avm-pipeline.git
cd avm-pipeline
mkdir build && cd build
cmake ..
make

# 5. Generate synthetic test data
cd ..
python3 scripts/generate_fisheye.py    # Per-camera calibration images
python3 scripts/generate_test_views.py  # 4-direction test views + GT extrinsics

# 6. Run
cd build
./pipeline
```

## Calibration System

### Intrinsic Calibration
Each of the 4 cameras is calibrated independently using chessboard images.
Calibration images are organized in per-camera directories:
```
data/calib/
├── front/    # Chessboard images for front camera
├── back/     # Chessboard images for back camera
├── left/     # Chessboard images for left camera
└── right/    # Chessboard images for right camera
```

The pipeline also supports a legacy single-directory mode (`data/calib_images/`)
where all cameras share one calibration — but per-camera is preferred.

### Extrinsic Calibration
Camera-to-vehicle extrinsic parameters (rotation R and translation t) are loaded
from `data/extrinsics_gt.json`. Each camera's pose is defined relative to the
vehicle body frame (X-right, Y-forward, Z-up).

The IPM homography is then derived mathematically:
```
H = K_cam · [r₁ | r₂ | -R·t] · H_bev
```
where r₁, r₂ are the first two columns of R (ground plane Z=0 constraint),
and H_bev maps BEV pixels to metric ground coordinates.

## Project Structure

```
avm-pipeline/
├── CMakeLists.txt
├── include/
│   ├── fisheye_calib.h        # Intrinsic calibration API
│   ├── extrinsic_calib.h      # Extrinsic calibration + homography API
│   ├── undistort.h            # Undistortion API (per-camera)
│   ├── birdview_stitch.h      # Stitching API (calib-derived IPM)
│   └── detect.h               # Detection API
├── src/
│   ├── fisheye_calib.cpp      # Per-camera fisheye calibration + RPE
│   ├── extrinsic_calib.cpp    # Extrinsic loading + homography computation
│   ├── undistort.cpp          # Per-camera undistortion
│   ├── birdview_stitch.cpp    # Calib-derived IPM + blending + stitching
│   ├── detect.cpp             # YOLOv8 ONNX inference + NMS
│   └── pipeline.cpp           # Main entry — chains all steps
├── data/
│   ├── calib/                 # Per-camera calibration images
│   │   ├── front/
│   │   ├── back/
│   │   ├── left/
│   │   └── right/
│   ├── test_views/            # 4-direction camera test images
│   ├── extrinsics_gt.json     # GT extrinsic parameters (R, t)
│   └── yolov8n.onnx           # YOLO model (not tracked in git)
├── output/                    # Generated results
└── scripts/
    ├── generate_fisheye.py    # Generate per-camera fisheye calibration images
    ├── generate_test_views.py # Generate test views + GT extrinsics JSON
    └── export_onnx.py         # One-time ONNX export script
```

## Output Files

| File | Description |
|------|-------------|
| `fisheye_calib_front.yaml` | Front camera intrinsic K + D (similarly for back/left/right) |
| `undistort_front.jpg` | Before/after undistortion comparison (per view) |
| `birdview.jpg` | Stitched bird's-eye view with feather blending |
| `birdview_no_blend.jpg` | Without blending (for comparison) |
| `birdview_detected.jpg` | Bird's-eye view with YOLO detection boxes |

## Limitations

- **Synthetic data**: The current test data is generated synthetically
  (`generate_fisheye.py` and `generate_test_views.py`), not captured from
  real fisheye cameras. The synthetic images use known GT parameters for
  validation purposes.
- **Extrinsics from config**: Extrinsic parameters are loaded from a JSON file
  rather than estimated from calibration targets on the ground. In a production
  system, these would be estimated using ground-plane calibration patterns
  (e.g., AprilTag boards at known positions).
- **Static blending**: Corner regions use simple 50/50 blending. A production
  system would use seam-finding or multi-band blending for smoother transitions.

## Environment

- **Language:** C++17
- **Libraries:** OpenCV 4.6.0, ONNX Runtime 1.16.3
- **Build:** CMake 3.10+
- **Platform:** Linux (WSL2 / Ubuntu 22.04)

## Related

- Fisheye camera calibration tool: [camera-calibration](https://github.com/jeason522/camera-calibration)
- YOLO C++ inference: [yolo-cpp-inference](https://github.com/jeason522/yolo-cpp-inference)
