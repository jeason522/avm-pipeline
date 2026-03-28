#include "birdview_stitch.h"
#include <iostream>
#include <cmath>

// 計算 feather blending 的權重 mask（離邊緣越近權重越低）
static cv::Mat createFeatherMask(int width, int height, int blendWidth) {
    cv::Mat mask(height, width, CV_32F, cv::Scalar(1.0f));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = std::min((float)x, (float)(width - 1 - x));
            float dy = std::min((float)y, (float)(height - 1 - y));
            float d = std::min(dx, dy);
            if (d < blendWidth) {
                mask.at<float>(y, x) = d / blendWidth;
            }
        }
    }
    return mask;
}

// 亮度補償：調整 src 使其平均亮度接近 target
static cv::Mat adjustBrightness(const cv::Mat& src, double targetMean) {
    cv::Scalar srcMean = cv::mean(src);
    double avgSrc = (srcMean[0] + srcMean[1] + srcMean[2]) / 3.0;
    if (avgSrc < 1.0) return src.clone();

    double gain = targetMean / avgSrc;
    gain = std::max(0.5, std::min(2.0, gain));  // 限制範圍

    cv::Mat adjusted;
    src.convertTo(adjusted, -1, gain, 0);
    return adjusted;
}

// 從校正參數推導的 IPM（Inverse Perspective Mapping）
// 使用 H = K_cam * [r0 | r1 | -R*t] 在地面平面 Z=0 上做投影
static cv::Mat applyCalibIPM(const cv::Mat& src,
                             const cv::Mat& K_cam,
                             const cv::Mat& R,
                             const cv::Mat& t_body,
                             cv::Size outSize) {
    // H_cam: 將地面座標 (X,Y) 投影到相機像素
    // P_cam = R * (P_body - t) = R*P_body - R*t
    // 因為 Z=0，只取 R 的前兩列
    cv::Mat Rt = -R * t_body;

    cv::Mat H_cam(3, 3, CV_64F);
    cv::Mat col0 = K_cam * R.col(0);
    cv::Mat col1 = K_cam * R.col(1);
    cv::Mat col2 = K_cam * Rt;
    col0.copyTo(H_cam.col(0));
    col1.copyTo(H_cam.col(1));
    col2.copyTo(H_cam.col(2));

    // BEV 像素 → 地面座標
    // 設定 BEV 範圍：以相機為中心，覆蓋 ±5m
    double metersPerPixel = 10.0 / outSize.width;  // 10m 寬度
    double cx_bev = outSize.width / 2.0;
    double cy_bev = outSize.height / 2.0;

    cv::Mat H_bev = (cv::Mat_<double>(3, 3) <<
        metersPerPixel, 0,               -cx_bev * metersPerPixel,
        0,              -metersPerPixel,  cy_bev * metersPerPixel,
        0,              0,                1.0);

    // H_total: BEV 像素 → 相機像素
    cv::Mat H = H_cam * H_bev;

    cv::Mat dst;
    cv::warpPerspective(src, dst, H, outSize,
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    return dst;
}

// Fallback: 硬編碼梯形 IPM（無校正參數時使用）
static cv::Mat applyFallbackIPM(const cv::Mat& src) {
    int W = src.cols, H = src.rows;

    std::vector<cv::Point2f> srcPts = {
        {W * 0.25f, 0},
        {W * 0.75f, 0},
        {(float)W,  (float)H},
        {0,         (float)H}
    };
    std::vector<cv::Point2f> dstPts = {
        {0,       0},
        {(float)W, 0},
        {(float)W, (float)H},
        {0,       (float)H}
    };

    cv::Mat H_mat = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat dst;
    cv::warpPerspective(src, dst, H_mat, {W, H});
    return dst;
}

// 繪製自車示意圖（俯視車體輪廓 + 方向箭頭）
static void drawEgoVehicle(cv::Mat& canvas, int W, int H) {
    int cx = W + W / 2;  // 車體中心 x
    int cy = H + H / 2;  // 車體中心 y

    int carW = W * 2 / 5;
    int carH = H * 3 / 4;
    int x1 = cx - carW / 2, y1 = cy - carH / 2;
    int x2 = cx + carW / 2, y2 = cy + carH / 2;

    // 填充車體底色
    cv::rectangle(canvas, {W, H}, {W * 2, H * 2}, {50, 50, 50}, -1);

    // 車體輪廓
    cv::rectangle(canvas, {x1, y1}, {x2, y2}, {160, 160, 160}, 2);

    // 擋風玻璃（前方梯形）
    int glassInset = carW / 6;
    std::vector<cv::Point> windshield = {
        {x1 + glassInset, y1 + 4},
        {x2 - glassInset, y1 + 4},
        {x2 - 2, y1 + carH / 5},
        {x1 + 2, y1 + carH / 5}
    };
    cv::polylines(canvas, windshield, true, {120, 180, 220}, 2);

    // 後擋玻璃
    std::vector<cv::Point> rearGlass = {
        {x1 + glassInset, y2 - 4},
        {x2 - glassInset, y2 - 4},
        {x2 - 2, y2 - carH / 6},
        {x1 + 2, y2 - carH / 6}
    };
    cv::polylines(canvas, rearGlass, true, {120, 180, 220}, 2);

    // 四個輪子
    int wheelW = carW / 8, wheelH = carH / 8;
    cv::rectangle(canvas, {x1 - wheelW, y1 + carH / 6}, {x1, y1 + carH / 6 + wheelH}, {140, 140, 140}, -1);
    cv::rectangle(canvas, {x2, y1 + carH / 6}, {x2 + wheelW, y1 + carH / 6 + wheelH}, {140, 140, 140}, -1);
    cv::rectangle(canvas, {x1 - wheelW, y2 - carH / 6 - wheelH}, {x1, y2 - carH / 6}, {140, 140, 140}, -1);
    cv::rectangle(canvas, {x2, y2 - carH / 6 - wheelH}, {x2 + wheelW, y2 - carH / 6}, {140, 140, 140}, -1);

    // 前進方向箭頭
    cv::arrowedLine(canvas, {cx, cy + carH / 6}, {cx, cy - carH / 4},
                    {0, 200, 0}, 3, cv::LINE_AA, 0, 0.3);

    // 方向標記
    double fontScale = std::min(W, H) / 600.0;
    cv::putText(canvas, "F", {cx - 8, y1 - 10},
        cv::FONT_HERSHEY_SIMPLEX, fontScale, {0, 200, 0}, 2);
}

cv::Mat stitchBirdView(const std::map<std::string, cv::Mat>& views,
                       const std::string& outputDir,
                       const std::map<std::string, CalibResult>& calibrations,
                       const std::map<std::string, CameraExtrinsics>& extrinsics) {
    // 檢查輸入
    if (views.find("front") == views.end() ||
        views.find("back")  == views.end() ||
        views.find("left")  == views.end() ||
        views.find("right") == views.end()) {
        std::cerr << "[stitch] 錯誤：需要 front/back/left/right 四張圖\n";
        return {};
    }

    int W = views.at("front").cols;
    int H = views.at("front").rows;

    // 判斷是否有完整的校正參數可以推導 Homography
    bool useCalibIPM = !calibrations.empty() && !extrinsics.empty();
    if (useCalibIPM) {
        for (auto& cam : {"front", "back", "left", "right"}) {
            if (calibrations.find(cam) == calibrations.end() ||
                extrinsics.find(cam) == extrinsics.end()) {
                useCalibIPM = false;
                break;
            }
        }
    }

    if (useCalibIPM) {
        std::cout << "[stitch] 使用校正參數推導 IPM Homography\n";
    } else {
        std::cout << "[stitch] 使用 fallback 梯形 IPM（無校正參數）\n";
    }

    // 1. 對每張圖做 IPM
    cv::Mat f, b, l, r;
    cv::Size bevTile(W, H);

    if (useCalibIPM) {
        f = applyCalibIPM(views.at("front"),
                          calibrations.at("front").K,
                          extrinsics.at("front").R,
                          extrinsics.at("front").t,
                          bevTile);
        b = applyCalibIPM(views.at("back"),
                          calibrations.at("back").K,
                          extrinsics.at("back").R,
                          extrinsics.at("back").t,
                          bevTile);
        l = applyCalibIPM(views.at("left"),
                          calibrations.at("left").K,
                          extrinsics.at("left").R,
                          extrinsics.at("left").t,
                          bevTile);
        r = applyCalibIPM(views.at("right"),
                          calibrations.at("right").K,
                          extrinsics.at("right").R,
                          extrinsics.at("right").t,
                          bevTile);
    } else {
        f = applyFallbackIPM(views.at("front"));
        b = applyFallbackIPM(views.at("back"));
        l = applyFallbackIPM(views.at("left"));
        r = applyFallbackIPM(views.at("right"));
    }

    // 2. 亮度補償（以 front 為基準）
    cv::Scalar frontMean = cv::mean(f);
    double targetBright = (frontMean[0] + frontMean[1] + frontMean[2]) / 3.0;
    b = adjustBrightness(b, targetBright);
    l = adjustBrightness(l, targetBright);
    r = adjustBrightness(r, targetBright);

    std::cout << "[stitch] 亮度補償完成 (target=" << targetBright << ")\n";

    // 3. 旋轉 left/right/back 到正確方向
    cv::Mat l_rot, r_rot, b_rot;
    cv::rotate(l, l_rot, cv::ROTATE_90_CLOCKWISE);
    cv::resize(l_rot, l_rot, {W, H});
    cv::rotate(r, r_rot, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::resize(r_rot, r_rot, {W, H});
    cv::rotate(b, b_rot, cv::ROTATE_180);

    // 4. 建立畫布（3x3 區塊）
    int outW = W * 3, outH = H * 3;
    cv::Mat canvas(outH, outW, CV_8UC3, cv::Scalar(40, 40, 40));

    // 5. 建立 feather mask
    int blendW = std::min(W, H) / 4;
    cv::Mat mask = createFeatherMask(W, H, blendW);
    cv::Mat mask3;
    cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3);

    // 6. 用 feather blending 貼到畫布
    auto blendPaste = [&](const cv::Mat& view, int x, int y) {
        cv::Rect roi(x, y, W, H);
        cv::Mat region = canvas(roi);
        cv::Mat viewF, regionF;
        view.convertTo(viewF, CV_32FC3);
        region.convertTo(regionF, CV_32FC3);

        cv::Scalar regionMean = cv::mean(region);
        if (regionMean[0] < 50 && regionMean[1] < 50 && regionMean[2] < 50) {
            view.copyTo(region);
        } else {
            cv::Mat blended = viewF.mul(mask3) + regionF.mul(cv::Scalar(1, 1, 1) - mask3);
            blended.convertTo(region, CV_8UC3);
        }
    };

    // 前：上方中間
    blendPaste(f, W, 0);
    // 後：下方中間
    blendPaste(b_rot, W, H * 2);
    // 左：中間左側
    blendPaste(l_rot, 0, H);
    // 右：中間右側
    blendPaste(r_rot, W * 2, H);

    // 7. 填充四角（用相鄰 view 的漸變延伸）
    // 左上角：混合 front 和 left
    {
        cv::Mat cornerF, cornerL;
        cv::resize(f(cv::Rect(0, H - H/2, W/2, H/2)), cornerF, {W, H});
        cv::resize(l_rot(cv::Rect(W - W/2, 0, W/2, H/2)), cornerL, {W, H});
        cv::Mat corner;
        cv::addWeighted(cornerF, 0.5, cornerL, 0.5, 0, corner);
        cv::GaussianBlur(corner, corner, {15, 15}, 5);
        corner.copyTo(canvas(cv::Rect(0, 0, W, H)));
    }
    // 右上角：混合 front 和 right
    {
        cv::Mat cornerF, cornerR;
        cv::resize(f(cv::Rect(W/2, H - H/2, W/2, H/2)), cornerF, {W, H});
        cv::resize(r_rot(cv::Rect(0, 0, W/2, H/2)), cornerR, {W, H});
        cv::Mat corner;
        cv::addWeighted(cornerF, 0.5, cornerR, 0.5, 0, corner);
        cv::GaussianBlur(corner, corner, {15, 15}, 5);
        corner.copyTo(canvas(cv::Rect(W * 2, 0, W, H)));
    }
    // 左下角：混合 back 和 left
    {
        cv::Mat cornerB, cornerL;
        cv::resize(b_rot(cv::Rect(0, 0, W/2, H/2)), cornerB, {W, H});
        cv::resize(l_rot(cv::Rect(W - W/2, H/2, W/2, H/2)), cornerL, {W, H});
        cv::Mat corner;
        cv::addWeighted(cornerB, 0.5, cornerL, 0.5, 0, corner);
        cv::GaussianBlur(corner, corner, {15, 15}, 5);
        corner.copyTo(canvas(cv::Rect(0, H * 2, W, H)));
    }
    // 右下角：混合 back 和 right
    {
        cv::Mat cornerB, cornerR;
        cv::resize(b_rot(cv::Rect(W/2, 0, W/2, H/2)), cornerB, {W, H});
        cv::resize(r_rot(cv::Rect(0, H/2, W/2, H/2)), cornerR, {W, H});
        cv::Mat corner;
        cv::addWeighted(cornerB, 0.5, cornerR, 0.5, 0, corner);
        cv::GaussianBlur(corner, corner, {15, 15}, 5);
        corner.copyTo(canvas(cv::Rect(W * 2, H * 2, W, H)));
    }

    // 8. 中間畫自車（ego vehicle）示意圖
    drawEgoVehicle(canvas, W, H);

    // 9. 儲存
    std::string outPath = outputDir + "/birdview.jpg";
    cv::imwrite(outPath, canvas);
    std::cout << "[stitch] 鳥瞰圖已儲存至 " << outPath << "\n";

    // 儲存無 blending 版本做對比
    cv::Mat canvasNB(outH, outW, CV_8UC3, cv::Scalar(40, 40, 40));
    f.copyTo(canvasNB(cv::Rect(W, 0, W, H)));
    b_rot.copyTo(canvasNB(cv::Rect(W, H * 2, W, H)));
    l_rot.copyTo(canvasNB(cv::Rect(0, H, W, H)));
    r_rot.copyTo(canvasNB(cv::Rect(W * 2, H, W, H)));
    drawEgoVehicle(canvasNB, W, H);
    cv::imwrite(outputDir + "/birdview_no_blend.jpg", canvasNB);
    std::cout << "[stitch] 無 blending 對比圖已儲存\n";

    return canvas;
}
