#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

// Per-camera extrinsic parameters: rotation and translation
// relative to the vehicle body frame.
//
// Vehicle body frame: X-right, Y-forward, Z-up, origin at ground center.
// Camera frame: x-right, y-down, z-forward (OpenCV convention).
//
// R: rotation from body frame to camera frame (3x3)
// t: camera position in body frame (3x1, meters)
struct CameraExtrinsics {
    cv::Mat R;   // 3x3 CV_64F
    cv::Mat t;   // 3x1 CV_64F
};

// Load extrinsic parameters for all 4 cameras from a JSON file.
// JSON format: { "front": {"R": [[...]], "t": [...]}, "back": ..., ... }
std::map<std::string, CameraExtrinsics> loadExtrinsics(
    const std::string& jsonPath);

// Save extrinsic parameters to a JSON file.
void saveExtrinsics(const std::string& jsonPath,
                    const std::map<std::string, CameraExtrinsics>& extrinsics);

// Compute the ground-plane homography from calibration parameters.
// Maps undistorted camera image pixels to bird's-eye view pixels.
//
// H = K_bev * [r1 r2 t_proj] * K_cam^{-1}
//
// where r1, r2 are the first two columns of R (body→cam),
// t_proj accounts for camera height and ground plane (z=0).
//
// Parameters:
//   K_cam:   3x3 camera intrinsic matrix
//   R:       3x3 rotation body→camera
//   t:       3x1 camera position in body frame (meters)
//   bevSize: output bird's-eye view size in pixels
//   metersPerPixel: scale factor (meters per pixel in BEV)
cv::Mat computeGroundHomography(const cv::Mat& K_cam,
                                const cv::Mat& R,
                                const cv::Mat& t,
                                cv::Size bevSize,
                                double metersPerPixel);
