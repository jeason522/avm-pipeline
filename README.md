# AVM Pipeline — Fisheye Calibration → Bird's-Eye Stitching → Object Detection

A complete C++ pipeline for automotive Around View Monitor (AVM) systems:
fisheye camera calibration, undistortion, bird's-eye view stitching with
feather blending, and YOLOv8 object detection — all in one executable.

## Pipeline

```
4 Fisheye Camera Images (front / back / left / right)
          │
          ▼
  ┌─────────────────────────┐
  │  Step 1: Fisheye Calib  │  cv::fisheye::calibrate()
  │  → K matrix + D coeffs  │  equidistant model (k1~k4)
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │  Step 2: Undistort      │  cv::fisheye::undistortImage()
  │  → 4 rectified views    │  before/after comparison
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │  Step 3: Bird's-Eye     │  Homography IPM
  │  Stitching + Blending   │  feather blending + gain compensation
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │  Step 4: YOLO Detect    │  YOLOv8n via ONNX Runtime (C++)
  │  → bounding boxes       │  NMS + 80-class COCO detection
  └──────────┬──────────────┘
             ▼
       output/result.jpg
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Fisheye calibration | `cv::fisheye::calibrate()` with per-image RPE analysis |
| Undistortion | `cv::fisheye::estimateNewCameraMatrixForUndistortRectify()` |
| IPM (Inverse Perspective Mapping) | Homography-based ground plane projection |
| Feather blending | Distance-weighted alpha masks for seamless stitching |
| Luminance compensation | Gain adjustment to match brightness across views |
| Corner filling | Blended interpolation from adjacent views |
| Object detection | YOLOv8n ONNX, hand-parsed (1,84,8400) tensor + NMS |

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

# 5. Prepare data
#    - Put chessboard images in data/calib_images/
#    - Put 4 camera views (front.jpg, back.jpg, left.jpg, right.jpg) in data/test_views/

# 6. Run
./pipeline
```

## Project Structure

```
avm-pipeline/
├── CMakeLists.txt
├── include/
│   ├── fisheye_calib.h        # Calibration API
│   ├── undistort.h            # Undistortion API
│   ├── birdview_stitch.h      # Stitching API
│   └── detect.h               # Detection API
├── src/
│   ├── fisheye_calib.cpp      # Fisheye calibration + per-image RPE
│   ├── undistort.cpp          # Fisheye undistortion + comparison output
│   ├── birdview_stitch.cpp    # IPM + blending + luminance compensation
│   ├── detect.cpp             # YOLOv8 ONNX inference + NMS
│   └── pipeline.cpp           # Main entry — chains all steps
├── data/
│   ├── calib_images/          # Chessboard calibration images
│   ├── test_views/            # 4-direction camera test images
│   └── yolov8n.onnx           # YOLO model (not tracked in git)
├── output/                    # Generated results
└── scripts/
    └── export_onnx.py         # One-time ONNX export script
```

## Output Files

| File | Description |
|------|-------------|
| `fisheye_calib.yaml` | Intrinsic matrix K + distortion coefficients D |
| `undistort_front.jpg` | Before/after undistortion comparison (per view) |
| `birdview.jpg` | Stitched bird's-eye view with feather blending |
| `birdview_no_blend.jpg` | Without blending (for comparison) |
| `birdview_detected.jpg` | Bird's-eye view with YOLO detection boxes |

## Environment

- **Language:** C++17
- **Libraries:** OpenCV 4.6.0, ONNX Runtime 1.16.3
- **Build:** CMake 3.10+
- **Platform:** Linux (WSL2 / Ubuntu 22.04)
