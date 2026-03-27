#include "undistort.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat fisheyeUndistort(const cv::Mat& src,
                         const cv::Mat& K, const cv::Mat& D) {
    cv::Mat newK;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, D, src.size(), cv::Matx33d::eye(), newK, 1.0);

    cv::Mat undistorted;
    cv::fisheye::undistortImage(src, undistorted, K, D, newK);
    return undistorted;
}

std::map<std::string, cv::Mat> undistortViews(
    const std::string& viewDir,
    const cv::Mat& K, const cv::Mat& D,
    const std::string& outputDir) {

    std::vector<std::string> names = {"front", "back", "left", "right"};
    std::map<std::string, cv::Mat> results;

    for (auto& name : names) {
        // 嘗試多種副檔名
        cv::Mat img;
        for (auto ext : {".jpg", ".png", ".jpeg"}) {
            std::string path = viewDir + "/" + name + ext;
            img = cv::imread(path);
            if (!img.empty()) break;
        }

        if (img.empty()) {
            std::cerr << "[undistort] 找不到 " << name << " 圖片\n";
            continue;
        }

        cv::Mat undistorted = fisheyeUndistort(img, K, D);
        results[name] = undistorted;

        // 儲存 before/after 對比
        cv::Mat comparison;
        cv::hconcat(img, undistorted, comparison);
        cv::putText(comparison, "Before", {20, 40},
            cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
        cv::putText(comparison, "After", {img.cols + 20, 40},
            cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

        std::string outPath = outputDir + "/undistort_" + name + ".jpg";
        cv::imwrite(outPath, comparison);
        std::cout << "[undistort] " << name << " → " << outPath << "\n";
    }

    return results;
}
