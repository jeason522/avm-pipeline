#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct CalibResult {
    cv::Mat K;          // 3x3 intrinsic matrix
    cv::Mat D;          // 4x1 distortion coefficients (k1~k4)
    double rpe;         // reprojection error
    cv::Size imageSize;
};

// Run fisheye calibration on chessboard images in imageDir.
// Saves result to outputYaml. Returns calibration result.
CalibResult fisheyeCalibrate(const std::string& imageDir,
                             const std::string& outputYaml,
                             int boardW = 9, int boardH = 6,
                             float squareSize = 25.0f);

// Load previously saved calibration from YAML.
CalibResult loadCalibration(const std::string& yamlPath);
