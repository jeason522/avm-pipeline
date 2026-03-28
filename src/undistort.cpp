#include "undistort.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat fisheyeUndistort(const cv::Mat& src,
                         const cv::Mat& K, const cv::Mat& D,
                         cv::Size calibSize) {
    cv::Mat scaledK = K.clone();

    // 如果校正時的圖片尺寸和輸入圖片不同，按比例縮放 K
    if (calibSize.width > 0 && calibSize.height > 0 &&
        (calibSize.width != src.cols || calibSize.height != src.rows)) {
        double sx = (double)src.cols / calibSize.width;
        double sy = (double)src.rows / calibSize.height;
        scaledK.at<double>(0, 0) *= sx;  // fx
        scaledK.at<double>(1, 1) *= sy;  // fy
        scaledK.at<double>(0, 2) *= sx;  // cx
        scaledK.at<double>(1, 2) *= sy;  // cy
        std::cout << "[undistort] K 已按比例縮放 ("
                  << calibSize << " → " << src.size() << ")\n";
    }

    cv::Mat newK;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        scaledK, D, src.size(), cv::Matx33d::eye(), newK, 1.0);

    cv::Mat undistorted;
    cv::fisheye::undistortImage(src, undistorted, scaledK, D, newK);
    return undistorted;
}

static cv::Mat loadView(const std::string& viewDir, const std::string& name) {
    cv::Mat img;
    for (auto ext : {".jpg", ".png", ".jpeg"}) {
        std::string path = viewDir + "/" + name + ext;
        img = cv::imread(path);
        if (!img.empty()) break;
    }
    if (img.empty()) {
        std::cerr << "[undistort] 找不到 " << name << " 圖片\n";
    }
    return img;
}

static void saveComparison(const cv::Mat& before, const cv::Mat& after,
                           const std::string& outputDir, const std::string& name) {
    cv::Mat comparison;
    cv::hconcat(before, after, comparison);
    cv::putText(comparison, "Before", {20, 40},
        cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
    cv::putText(comparison, "After", {before.cols + 20, 40},
        cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

    std::string outPath = outputDir + "/undistort_" + name + ".jpg";
    cv::imwrite(outPath, comparison);
    std::cout << "[undistort] " << name << " → " << outPath << "\n";
}

std::map<std::string, cv::Mat> undistortViews(
    const std::string& viewDir,
    const cv::Mat& K, const cv::Mat& D,
    const std::string& outputDir,
    cv::Size calibSize) {

    std::vector<std::string> names = {"front", "back", "left", "right"};
    std::map<std::string, cv::Mat> results;

    for (auto& name : names) {
        cv::Mat img = loadView(viewDir, name);
        if (img.empty()) continue;

        cv::Mat undistorted = fisheyeUndistort(img, K, D, calibSize);
        results[name] = undistorted;
        saveComparison(img, undistorted, outputDir, name);
    }

    return results;
}

std::map<std::string, cv::Mat> undistortViewsPerCamera(
    const std::string& viewDir,
    const std::map<std::string, CalibResult>& calibrations,
    const std::string& outputDir) {

    std::vector<std::string> names = {"front", "back", "left", "right"};
    std::map<std::string, cv::Mat> results;

    for (auto& name : names) {
        auto it = calibrations.find(name);
        if (it == calibrations.end() || it->second.K.empty()) {
            std::cerr << "[undistort] 找不到 " << name << " 的校正參數\n";
            continue;
        }

        cv::Mat img = loadView(viewDir, name);
        if (img.empty()) continue;

        const auto& calib = it->second;
        cv::Mat undistorted = fisheyeUndistort(img, calib.K, calib.D, calib.imageSize);
        results[name] = undistorted;
        saveComparison(img, undistorted, outputDir, name);
    }

    return results;
}
