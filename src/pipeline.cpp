#include "fisheye_calib.h"
#include "undistort.h"
#include "birdview_stitch.h"
#include "detect.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::cout << "=== AVM Pipeline ===\n\n";

    std::string dataDir   = "../data";
    std::string outputDir = "../output";
    std::string calibYaml = outputDir + "/fisheye_calib.yaml";
    std::string modelPath = dataDir + "/yolov8n.onnx";

    // 確保 output 目錄存在
    fs::create_directories(outputDir);

    // ─── Step 1: 校正（或載入已有結果）───────────────────
    CalibResult calib;
    if (fs::exists(calibYaml)) {
        std::cout << "[Step 1] 載入已有的校正結果...\n";
        calib = loadCalibration(calibYaml);
    } else {
        std::string calibDir = dataDir + "/calib_images";
        if (!fs::exists(calibDir) || fs::is_empty(calibDir)) {
            std::cerr << "[Step 1] 錯誤：找不到校正圖片 (" << calibDir << ")\n";
            std::cerr << "  請放入棋盤格圖片後再執行。\n";
            std::cerr << "  如果已有校正結果，請放到 " << calibYaml << "\n";
            return 1;
        }
        std::cout << "[Step 1] 執行魚眼校正...\n";
        calib = fisheyeCalibrate(calibDir, calibYaml);
    }

    if (calib.K.empty()) {
        std::cerr << "校正失敗，無法繼續\n";
        return 1;
    }
    std::cout << "[Step 1] 完成 (RPE=" << calib.rpe << " px)\n\n";

    // ─── Step 2: 去畸變 ─────────────────────────────────
    std::string viewDir = dataDir + "/test_views";
    if (!fs::exists(viewDir)) {
        std::cerr << "[Step 2] 錯誤：找不到測試圖片 (" << viewDir << ")\n";
        return 1;
    }

    std::cout << "[Step 2] 去畸變...\n";
    auto views = undistortViews(viewDir, calib.K, calib.D, outputDir);

    if (views.size() < 4) {
        std::cerr << "[Step 2] 警告：只找到 " << views.size() << "/4 張圖片\n";
        if (views.empty()) return 1;
    }
    std::cout << "[Step 2] 完成\n\n";

    // ─── Step 3: 鳥瞰拼接 ──────────────────────────────
    std::cout << "[Step 3] 拼接鳥瞰圖...\n";
    cv::Mat birdview = stitchBirdView(views, outputDir);

    if (birdview.empty()) {
        std::cerr << "[Step 3] 拼接失敗\n";
        return 1;
    }
    std::cout << "[Step 3] 完成\n\n";

    // ─── Step 4: YOLO 偵測 ──────────────────────────────
    if (fs::exists(modelPath)) {
        std::cout << "[Step 4] YOLO 偵測...\n";
        cv::Mat detected = detectObjects(birdview, modelPath);
        cv::imwrite(outputDir + "/birdview_detected.jpg", detected);
        std::cout << "[Step 4] 完成 → output/birdview_detected.jpg\n\n";
    } else {
        std::cout << "[Step 4] 跳過（找不到模型 " << modelPath << "）\n";
        std::cout << "  請下載 yolov8n.onnx 放到 data/ 目錄\n\n";
    }

    std::cout << "=== Pipeline 完成 ===\n";
    std::cout << "輸出檔案在 output/ 目錄中：\n";
    std::cout << "  fisheye_calib.yaml     — 校正結果\n";
    std::cout << "  undistort_*.jpg        — 去畸變對比\n";
    std::cout << "  birdview.jpg           — 鳥瞰拼接圖\n";
    std::cout << "  birdview_no_blend.jpg  — 無 blending 對比\n";
    std::cout << "  birdview_detected.jpg  — YOLO 偵測結果\n";

    return 0;
}
