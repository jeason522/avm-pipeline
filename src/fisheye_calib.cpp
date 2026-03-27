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

    if (successCount < 4) {
        std::cerr << "[calib] 錯誤：至少需要 4 張成功偵測的圖片。\n";
        return {};
    }

    // 3. 執行魚眼校正
    std::cout << "[calib] 執行魚眼校正...\n";
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                cv::fisheye::CALIB_FIX_SKEW;

    double rpe = cv::fisheye::calibrate(objPoints, imgPoints, imgSize,
                                        K, D, rvecs, tvecs, flags);

    std::cout << "[calib] 重投影誤差 (RPE): " << rpe << " px\n";
    std::cout << "[calib] 內參矩陣 K:\n" << K << "\n";
    std::cout << "[calib] 畸變係數 D (k1~k4):\n" << D << "\n";

    // 4. Per-image RPE 分析
    std::cout << "\n[calib] Per-image RPE:\n";
    for (size_t i = 0; i < objPoints.size(); i++) {
        std::vector<cv::Point2f> projected;
        cv::fisheye::projectPoints(objPoints[i], projected, rvecs[i], tvecs[i], K, D);
        double err = 0;
        for (size_t j = 0; j < projected.size(); j++) {
            cv::Point2f diff = projected[j] - imgPoints[i][j];
            err += std::sqrt(diff.x * diff.x + diff.y * diff.y);
        }
        err /= projected.size();
        std::cout << "  Image " << i << ": " << err << " px\n";
    }

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

#ifdef STANDALONE_CALIB
int main() {
    auto result = fisheyeCalibrate("../data/calib_images",
                                   "../output/fisheye_calib.yaml");
    if (result.K.empty()) {
        std::cerr << "校正失敗\n";
        return 1;
    }

    // 輸出一張校正前後對比圖
    namespace fs = std::filesystem;
    for (auto& entry : fs::directory_iterator("../data/calib_images")) {
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
            cv::imwrite("../output/calib_comparison.jpg", comparison);
            std::cout << "對比圖已儲存至 output/calib_comparison.jpg\n";
            break;
        }
    }
    return 0;
}
#endif
