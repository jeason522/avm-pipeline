#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include "fisheye_calib.h"
#include "extrinsic_calib.h"

// Stitch 4 undistorted views into a bird's-eye view.
// Uses calibration-derived homography (from K, R, t) for IPM projection.
// Falls back to hardcoded trapezoid IPM if no extrinsics are provided.
cv::Mat stitchBirdView(const std::map<std::string, cv::Mat>& views,
                       const std::string& outputDir,
                       const std::map<std::string, CalibResult>& calibrations = {},
                       const std::map<std::string, CameraExtrinsics>& extrinsics = {});
