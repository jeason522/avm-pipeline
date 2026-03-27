#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// Run YOLOv8 detection on input image using ONNX Runtime.
// Returns image with bounding boxes drawn.
cv::Mat detectObjects(const cv::Mat& input,
                      const std::string& modelPath,
                      float confThr = 0.25f,
                      float iouThr = 0.45f);
