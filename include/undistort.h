#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

// Undistort a single image using fisheye model.
cv::Mat fisheyeUndistort(const cv::Mat& src,
                         const cv::Mat& K, const cv::Mat& D);

// Undistort 4 views (front/back/left/right) and return them.
// Also saves before/after comparison images to outputDir.
std::map<std::string, cv::Mat> undistortViews(
    const std::string& viewDir,
    const cv::Mat& K, const cv::Mat& D,
    const std::string& outputDir);
