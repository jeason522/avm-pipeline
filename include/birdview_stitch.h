#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>

// Stitch 4 undistorted views into a bird's-eye view.
// Applies IPM (Inverse Perspective Mapping), rotation alignment,
// feather blending, and luminance compensation.
cv::Mat stitchBirdView(const std::map<std::string, cv::Mat>& views,
                       const std::string& outputDir);
