#include "extrinsic_calib.h"
#include <fstream>
#include <iostream>
#include <sstream>

// Minimal JSON parsing for our specific format.
// We avoid external dependencies (nlohmann/json) to keep the build simple.

static std::string readFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Parse a JSON array of numbers like [1.0, 2.0, 3.0]
static std::vector<double> parseNumberArray(const std::string& s,
                                            size_t& pos) {
    std::vector<double> nums;
    // Find opening bracket
    pos = s.find('[', pos);
    if (pos == std::string::npos) return nums;
    pos++; // skip '['

    while (pos < s.size()) {
        // Skip whitespace
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' ||
               s[pos] == '\r' || s[pos] == '\t')) pos++;

        if (s[pos] == ']') { pos++; break; }
        if (s[pos] == ',') { pos++; continue; }

        // Check for nested array
        if (s[pos] == '[') {
            auto inner = parseNumberArray(s, pos);
            nums.insert(nums.end(), inner.begin(), inner.end());
            continue;
        }

        // Parse number
        size_t end;
        double val = std::stod(s.substr(pos), &end);
        nums.push_back(val);
        pos += end;
    }
    return nums;
}

// Find "key": in JSON string starting from pos
static size_t findKey(const std::string& s, const std::string& key,
                      size_t start = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = s.find(pattern, start);
    if (pos != std::string::npos) {
        pos += pattern.size();
        // Skip whitespace and colon
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ':' ||
               s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;
    }
    return pos;
}

std::map<std::string, CameraExtrinsics> loadExtrinsics(
    const std::string& jsonPath) {

    std::map<std::string, CameraExtrinsics> result;
    std::string content = readFile(jsonPath);

    if (content.empty()) {
        std::cerr << "[extrinsic] 錯誤：無法讀取 " << jsonPath << "\n";
        return result;
    }

    std::vector<std::string> cameras = {"front", "back", "left", "right"};

    for (auto& cam : cameras) {
        size_t camPos = findKey(content, cam);
        if (camPos == std::string::npos) {
            std::cerr << "[extrinsic] 警告：找不到 " << cam << " 的外參\n";
            continue;
        }

        // Find R
        size_t rPos = findKey(content, "R", camPos);
        if (rPos == std::string::npos) continue;
        auto rVals = parseNumberArray(content, rPos);

        // Find t
        size_t tPos = findKey(content, "t", camPos);
        if (tPos == std::string::npos) continue;
        auto tVals = parseNumberArray(content, tPos);

        if (rVals.size() != 9 || tVals.size() != 3) {
            std::cerr << "[extrinsic] 警告：" << cam
                      << " 的參數維度不正確 (R=" << rVals.size()
                      << ", t=" << tVals.size() << ")\n";
            continue;
        }

        CameraExtrinsics ext;
        ext.R = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 9; i++)
            ext.R.at<double>(i / 3, i % 3) = rVals[i];

        ext.t = cv::Mat(3, 1, CV_64F);
        for (int i = 0; i < 3; i++)
            ext.t.at<double>(i) = tVals[i];

        result[cam] = ext;
        std::cout << "[extrinsic] " << cam << ": t=["
                  << tVals[0] << ", " << tVals[1] << ", " << tVals[2]
                  << "] m\n";
    }

    return result;
}

void saveExtrinsics(const std::string& jsonPath,
                    const std::map<std::string, CameraExtrinsics>& extrinsics) {
    std::ofstream ofs(jsonPath);
    ofs << "{\n";
    bool first = true;
    for (auto& [name, ext] : extrinsics) {
        if (!first) ofs << ",\n";
        first = false;

        ofs << "  \"" << name << "\": {\n";
        ofs << "    \"R\": [";
        for (int i = 0; i < 3; i++) {
            if (i > 0) ofs << ", ";
            ofs << "[";
            for (int j = 0; j < 3; j++) {
                if (j > 0) ofs << ", ";
                ofs << ext.R.at<double>(i, j);
            }
            ofs << "]";
        }
        ofs << "],\n";

        ofs << "    \"t\": [";
        for (int i = 0; i < 3; i++) {
            if (i > 0) ofs << ", ";
            ofs << ext.t.at<double>(i);
        }
        ofs << "]\n";
        ofs << "  }";
    }
    ofs << "\n}\n";
    ofs.close();
    std::cout << "[extrinsic] 外參已儲存至 " << jsonPath << "\n";
}

cv::Mat computeGroundHomography(const cv::Mat& K_cam,
                                const cv::Mat& R,
                                const cv::Mat& t,
                                cv::Size bevSize,
                                double metersPerPixel) {
    // The homography maps points on the ground plane (Z=0 in body frame)
    // to camera image pixels.
    //
    // A 3D point on the ground: P_body = [X, Y, 0]^T
    // In camera frame: P_cam = R * (P_body - t_cam)
    //                        = R * P_body - R * t_cam
    //
    // Since Z=0, we only need columns 0,1 of R for X,Y:
    //   P_cam = [r0 r1] * [X; Y] + (-R * t)
    //
    // Projected to image: p = K * P_cam (homogeneous)
    //   p ~ K * [r0 | r1 | -R*t] * [X; Y; 1]
    //
    // This is the camera homography H_cam = K * [r0 | r1 | -R*t]
    //
    // For BEV, we define a virtual top-down camera:
    //   BEV pixel (u, v) maps to body frame:
    //     X = (u - bevSize.width/2) * metersPerPixel
    //     Y = (bevSize.height/2 - v) * metersPerPixel  (Y forward = up in image)
    //
    // So: H_total = H_cam * H_bev^{-1}
    // where H_bev maps BEV pixels to body ground coordinates.

    // Step 1: Camera homography H_cam = K * [r0 | r1 | -R*t]
    cv::Mat Rt = -R * t;  // translation in camera frame

    cv::Mat H_cam(3, 3, CV_64F);
    // Column 0: K * r0
    cv::Mat r0 = R.col(0);
    cv::Mat r1 = R.col(1);
    cv::Mat col0 = K_cam * r0;
    cv::Mat col1 = K_cam * r1;
    cv::Mat col2 = K_cam * Rt;

    col0.copyTo(H_cam.col(0));
    col1.copyTo(H_cam.col(1));
    col2.copyTo(H_cam.col(2));

    // Step 2: BEV pixel → body ground coordinate transform
    // X = (u - cx_bev) * mpp
    // Y = (cy_bev - v) * mpp
    // In matrix form: [X; Y; 1] = H_bev * [u; v; 1]
    double cx_bev = bevSize.width / 2.0;
    double cy_bev = bevSize.height / 2.0;

    cv::Mat H_bev = (cv::Mat_<double>(3, 3) <<
        metersPerPixel, 0,               -cx_bev * metersPerPixel,
        0,              -metersPerPixel,  cy_bev * metersPerPixel,
        0,              0,                1.0);

    // Step 3: Combined homography
    // Maps BEV pixel → body ground → camera image
    // H = H_cam * H_bev
    cv::Mat H = H_cam * H_bev;

    // We want the inverse: camera image → BEV pixel
    // But for warpPerspective(src=camera, dst=BEV), we need the
    // mapping from BEV pixel to camera pixel, which is H itself.
    // Actually warpPerspective maps dst←src, so we need H that maps
    // BEV coords to camera coords. That's H = H_cam * H_bev.
    // But we use it as: warpPerspective(camera_img, bev_img, H_inv, bevSize)
    // where H_inv = inv(H_cam * H_bev) maps camera → BEV.
    // Or: warpPerspective with WARP_INVERSE_MAP flag using H directly.

    return H;  // BEV pixel → camera pixel (use with WARP_INVERSE_MAP)
}
