#include "fisheye_calib.h"
#include "undistort.h"
#include "birdview_stitch.h"
#include "detect.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --skip-calib    Skip calibration/undistortion, use raw images directly\n"
              << "  --help          Show this help\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== AVM Pipeline ===\n\n";

    bool skipCalib = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--skip-calib") skipCalib = true;
        else if (arg == "--help") { printUsage(argv[0]); return 0; }
    }

    std::string dataDir   = "../data";
    std::string outputDir = "../output";
    std::string calibYaml = outputDir + "/fisheye_calib.yaml";
    std::string modelPath = dataDir + "/yolov8n.onnx";
    std::string viewDir   = dataDir + "/test_views";

    // 確保 output 目錄存在
    fs::create_directories(outputDir);

    std::map<std::string, cv::Mat> views;

    if (skipCalib) {
        // ─── 跳過校正模式：直接讀原圖 ────────────────────
        std::cout << "[Mode] 跳過校正，直接使用原圖\n\n";

        std::vector<std::string> names = {"front", "back", "left", "right"};
        for (auto& name : names) {
            cv::Mat img;
            for (auto ext : {".jpg", ".png", ".jpeg"}) {
                std::string path = viewDir + "/" + name + ext;
                img = cv::imread(path);
                if (!img.empty()) break;
            }
            if (img.empty()) {
                std::cerr << "[load] 找不到 " << name << " 圖片\n";
            } else {
                views[name] = img;
                std::cout << "[load] " << name << " (" << img.cols << "x" << img.rows << ")\n";
            }
        }

        if (views.size() < 4) {
            std::cerr << "錯誤：需要 4 張圖片\n";
            return 1;
        }
        std::cout << "\n";

    } else {
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
                std::cerr << "  或使用 --skip-calib 跳過校正步驟。\n";
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
        if (!fs::exists(viewDir)) {
            std::cerr << "[Step 2] 錯誤：找不到測試圖片 (" << viewDir << ")\n";
            return 1;
        }

        std::cout << "[Step 2] 去畸變...\n";
        views = undistortViews(viewDir, calib.K, calib.D, outputDir);

        if (views.size() < 4) {
            std::cerr << "[Step 2] 警告：只找到 " << views.size() << "/4 張圖片\n";
            if (views.empty()) return 1;
        }
        std::cout << "[Step 2] 完成\n\n";
    }

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
    if (!skipCalib) {
        std::cout << "  fisheye_calib.yaml     — 校正結果\n";
        std::cout << "  undistort_*.jpg        — 去畸變對比\n";
    }
    std::cout << "  birdview.jpg           — 鳥瞰拼接圖\n";
    std::cout << "  birdview_no_blend.jpg  — 無 blending 對比\n";
    std::cout << "  birdview_detected.jpg  — YOLO 偵測結果\n";

    return 0;
}
