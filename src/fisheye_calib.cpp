#include "fisheye_calib.h"
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

CalibResult fisheyeCalibrate(const std::string& imageDir,
                             const std::string& outputYaml,
                             int boardW, int boardH,
                             float squareSize) {
    cv::Size boardSize(boardW, boardH);

    // 1. 建立理想 3D 角點座標（Z=0 平面）
    std::vector<cv::Point3f> objTemplate;
    for (int r = 0; r < boardH; r++)
        for (int c = 0; c < boardW; c++)
            objTemplate.emplace_back(c * squareSize, r * squareSize, 0.0f);

    std::vector<std::vector<cv::Point3f>> objPoints;
    std::vector<std::vector<cv::Point2f>> imgPoints;

    // 2. 讀取所有圖片，找角點
    std::vector<std::string> paths;
    for (auto& entry : fs::directory_iterator(imageDir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());

    if (paths.empty()) {
        std::cerr << "[calib] 錯誤：" << imageDir << " 裡沒有圖片！\n";
        return {};
    }

    std::cout << "[calib] 找到 " << paths.size() << " 張圖片，開始偵測角點...\n";

    cv::Size imgSize;
    int successCount = 0;

    for (auto& path : paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        imgSize = img.size();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                 30, 0.001));
            objPoints.push_back(objTemplate);
            imgPoints.push_back(corners);
            successCount++;
            std::cout << "  ✓ " << fs::path(path).filename().string() << "\n";
        } else {
            std::cout << "  ✗ " << fs::path(path).filename().string() << "\n";
        }
    }

    std::cout << "[calib] 成功偵測：" << successCount << " / " << paths.size() << "\n";

    if (successCount < 3) {
        std::cerr << "[calib] 錯誤：至少需要 3 張成功偵測的圖片。\n";
        return {};
    }

    // 3. 執行魚眼校正（含錯誤處理）
    std::cout << "[calib] 執行魚眼校正...\n";
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                cv::fisheye::CALIB_FIX_SKEW;

    double rpe = -1;
    // OpenCV fisheye::calibrate 可能在某些角點配置下崩潰
    // 嘗試用所有圖片；如果失敗，逐一排除問題圖片再試
    try {
        rpe = cv::fisheye::calibrate(objPoints, imgPoints, imgSize,
                                     K, D, rvecs, tvecs, flags);
    } catch (const cv::Exception& e) {
        std::cout << "[calib] 初次校正失敗，嘗試排除問題圖片...\n";
        // 逐一排除每張圖片嘗試
        for (size_t skip = 0; skip < objPoints.size(); skip++) {
            auto objTmp = objPoints;
            auto imgTmp = imgPoints;
            objTmp.erase(objTmp.begin() + skip);
            imgTmp.erase(imgTmp.begin() + skip);

            if (objTmp.size() < 3) continue;

            K = cv::Mat::eye(3, 3, CV_64F);
            D = cv::Mat::zeros(4, 1, CV_64F);
            rvecs.clear();
            tvecs.clear();

            try {
                rpe = cv::fisheye::calibrate(objTmp, imgTmp, imgSize,
                                             K, D, rvecs, tvecs, flags);
                std::cout << "[calib] 排除 Image " << skip
                          << " 後校正成功 (RPE=" << rpe << ")\n";
                // 更新 points（後面 per-image RPE 用）
                objPoints = objTmp;
                imgPoints = imgTmp;
                break;
            } catch (...) {
                continue;
            }
        }
    }

    if (rpe < 0) {
        std::cerr << "[calib] 錯誤：魚眼校正失敗\n";
        return {};
    }

    std::cout << "[calib] 重投影誤差 (RPE): " << rpe << " px\n";
    std::cout << "[calib] 內參矩陣 K:\n" << K << "\n";
    std::cout << "[calib] 畸變係數 D (k1~k4):\n" << D << "\n";

    // 4. Per-image RPE 分析 + 自動排除離群圖片
    const double RPE_OUTLIER_THRESHOLD = 50.0;  // px
    std::cout << "\n[calib] Per-image RPE:\n";
    std::vector<size_t> outlierIndices;
    for (size_t i = 0; i < objPoints.size(); i++) {
        std::vector<cv::Point2f> projected;
        cv::fisheye::projectPoints(objPoints[i], projected, rvecs[i], tvecs[i], K, D);
        double err = 0;
        for (size_t j = 0; j < projected.size(); j++) {
            cv::Point2f diff = projected[j] - imgPoints[i][j];
            err += std::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
        err /= projected.size();
        std::cout << "  Image " << i << ": " << err << " px";
        if (err > RPE_OUTLIER_THRESHOLD) {
            std::cout << "  ← outlier";
            outlierIndices.push_back(i);
        }
        std::cout << "\n";
    }

    // 如果有離群圖片且排除後仍有 >= 3 張，重新校正
    if (!outlierIndices.empty() &&
        (objPoints.size() - outlierIndices.size()) >= 3) {
        std::cout << "[calib] 排除 " << outlierIndices.size()
                  << " 張離群圖片，重新校正...\n";

        // 從後往前刪除（避免 index 位移）
        auto objClean = objPoints;
        auto imgClean = imgPoints;
        for (int idx = (int)outlierIndices.size() - 1; idx >= 0; idx--) {
            objClean.erase(objClean.begin() + outlierIndices[idx]);
            imgClean.erase(imgClean.begin() + outlierIndices[idx]);
        }

        cv::Mat K2 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat D2 = cv::Mat::zeros(4, 1, CV_64F);
        std::vector<cv::Mat> rvecs2, tvecs2;

        try {
            double rpe2 = cv::fisheye::calibrate(objClean, imgClean, imgSize,
                                                  K2, D2, rvecs2, tvecs2, flags);
            std::cout << "[calib] 重新校正 RPE: " << rpe2 << " px (原 " << rpe << " px)\n";
            // 採用新結果
            K = K2; D = D2; rpe = rpe2;
            rvecs = rvecs2; tvecs = tvecs2;
            objPoints = objClean; imgPoints = imgClean;

            // 印出新的 per-image RPE
            std::cout << "[calib] 新 Per-image RPE:\n";
            for (size_t i = 0; i < objPoints.size(); i++) {
                std::vector<cv::Point2f> projected;
                cv::fisheye::projectPoints(objPoints[i], projected,
                                           rvecs[i], tvecs[i], K, D);
                double err = 0;
                for (size_t j = 0; j < projected.size(); j++) {
                    cv::Point2f diff = projected[j] - imgPoints[i][j];
                    err += std::sqrt(diff.x * diff.x + diff.y * diff.y);
                }
                err /= projected.size();
                std::cout << "  Image " << i << ": " << err << " px\n";
            }
        } catch (const cv::Exception& e) {
            std::cout << "[calib] 重新校正失敗，保留原結果\n";
        }
    }

    std::cout << "\n[calib] 最終 K:\n" << K << "\n";
    std::cout << "[calib] 最終 D:\n" << D << "\n";
    std::cout << "[calib] 最終 RPE: " << rpe << " px\n";

    // 5. 儲存為 YAML
    cv::FileStorage fs_out(outputYaml, cv::FileStorage::WRITE);
    fs_out << "image_width" << imgSize.width;
    fs_out << "image_height" << imgSize.height;
    fs_out << "camera_matrix" << K;
    fs_out << "dist_coeffs" << D;
    fs_out << "reprojection_error" << rpe;
    fs_out.release();
    std::cout << "[calib] 結果已儲存至 " << outputYaml << "\n";

    return {K, D, rpe, imgSize};
}

CalibResult loadCalibration(const std::string& yamlPath) {
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[calib] 錯誤：無法開啟 " << yamlPath << "\n";
        return {};
    }

    CalibResult result;
    fs["camera_matrix"] >> result.K;
    fs["dist_coeffs"] >> result.D;
    fs["reprojection_error"] >> result.rpe;

    int w, h;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    result.imageSize = cv::Size(w, h);
    fs.release();

    std::cout << "[calib] 已載入校正結果 (RPE=" << result.rpe << ")\n";
    return result;
}

std::map<std::string, CalibResult> calibrateAllCameras(
    const std::string& calibDir,
    const std::string& outputDir,
    int boardW, int boardH,
    float squareSize) {

    std::map<std::string, CalibResult> results;
    std::vector<std::string> cameras = {"front", "back", "left", "right"};

    // Check if per-camera subdirectories exist
    bool hasSubDirs = true;
    for (auto& cam : cameras) {
        if (!fs::exists(calibDir + "/" + cam)) {
            hasSubDirs = false;
            break;
        }
    }

    if (hasSubDirs) {
        // Per-camera independent calibration
        std::cout << "[calib] 偵測到 4 顆相機的校正圖目錄，執行獨立校正...\n\n";
        for (auto& cam : cameras) {
            std::string camDir = calibDir + "/" + cam;
            std::string yamlPath = outputDir + "/fisheye_calib_" + cam + ".yaml";

            std::cout << "── " << cam << " 相機 ──\n";
            auto result = fisheyeCalibrate(camDir, yamlPath, boardW, boardH, squareSize);
            if (!result.K.empty()) {
                results[cam] = result;
            } else {
                std::cerr << "[calib] 警告：" << cam << " 校正失敗\n";
            }
            std::cout << "\n";
        }
    } else {
        // Fallback: single calibration for all cameras
        std::cout << "[calib] 未偵測到分目錄，使用單一校正（所有相機共用）...\n";
        std::string yamlPath = outputDir + "/fisheye_calib.yaml";
        auto result = fisheyeCalibrate(calibDir, yamlPath, boardW, boardH, squareSize);
        if (!result.K.empty()) {
            for (auto& cam : cameras) {
                results[cam] = result;
            }
        }
    }

    return results;
}

std::map<std::string, CalibResult> loadAllCalibrations(
    const std::string& outputDir) {

    std::map<std::string, CalibResult> results;
    std::vector<std::string> cameras = {"front", "back", "left", "right"};

    // Try per-camera files first
    bool hasPerCamera = true;
    for (auto& cam : cameras) {
        std::string path = outputDir + "/fisheye_calib_" + cam + ".yaml";
        if (!fs::exists(path)) {
            hasPerCamera = false;
            break;
        }
    }

    if (hasPerCamera) {
        std::cout << "[calib] 載入 4 顆相機的獨立校正結果...\n";
        for (auto& cam : cameras) {
            std::string path = outputDir + "/fisheye_calib_" + cam + ".yaml";
            std::cout << "  " << cam << ": ";
            results[cam] = loadCalibration(path);
        }
    } else {
        // Fallback: single calibration file
        std::string path = outputDir + "/fisheye_calib.yaml";
        if (fs::exists(path)) {
            std::cout << "[calib] 載入共用校正結果...\n";
            auto result = loadCalibration(path);
            for (auto& cam : cameras) {
                results[cam] = result;
            }
        }
    }

    return results;
}

#ifdef STANDALONE_CALIB
int main(int argc, char* argv[]) {
    std::string calibDir = "../data/calib";
    std::string outputDir = "../output";

    // Check for per-camera or legacy directory
    if (!fs::exists(calibDir)) {
        calibDir = "../data/calib_images";
    }

    fs::create_directories(outputDir);

    auto results = calibrateAllCameras(calibDir, outputDir);

    if (results.empty()) {
        std::cerr << "校正失敗\n";
        return 1;
    }

    // 輸出校正前後對比圖
    for (auto& [cam, result] : results) {
        std::string imgDir = "../data/calib/" + cam;
        if (!fs::exists(imgDir)) imgDir = "../data/calib_images";

        for (auto& entry : fs::directory_iterator(imgDir)) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png") {
                cv::Mat sample = cv::imread(entry.path().string());
                cv::Mat undistorted;
                cv::fisheye::undistortImage(sample, undistorted, result.K, result.D);

                cv::Mat comparison;
                cv::hconcat(sample, undistorted, comparison);
                cv::putText(comparison, "Before", {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
                cv::putText(comparison, "After (undistorted)",
                    {sample.cols + 20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
                cv::imwrite(outputDir + "/calib_comparison_" + cam + ".jpg", comparison);
                std::cout << cam << " 對比圖已儲存\n";
                break;
            }
        }
    }
    return 0;
}
#endif
