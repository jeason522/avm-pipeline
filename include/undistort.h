#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include "fisheye_calib.h"

// Undistort a single image using fisheye model.
cv::Mat fisheyeUndistort(const cv::Mat& src,
                         const cv::Mat& K, const cv::Mat& D,
                         cv::Size calibSize = {});

// Undistort 4 views using a single shared K, D.
// Also saves before/after comparison images to outputDir.
std::map<std::string, cv::Mat> undistortViews(
    const std::string& viewDir,
    const cv::Mat& K, const cv::Mat& D,
    const std::string& outputDir,
    cv::Size calibSize = {});

// Undistort 4 views using per-camera K, D from calibration results.
std::map<std::string, cv::Mat> undistortViewsPerCamera(
    const std::string& viewDir,
    const std::map<std::string, CalibResult>& calibrations,
    const std::string& outputDir);
