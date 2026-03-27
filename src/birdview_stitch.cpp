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

// IPM: 把透視圖投影到俯視圖（Inverse Perspective Mapping）
static cv::Mat applyIPM(const cv::Mat& src) {
    int W = src.cols, H = src.rows;

    // 原圖中的梯形區域（模擬透視效果）
    std::vector<cv::Point2f> srcPts = {
        {W * 0.25f, 0},
        {W * 0.75f, 0},
        {(float)W,  (float)H},
        {0,         (float)H}
    };
    // 目標：拉成矩形（俯視）
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

cv::Mat stitchBirdView(const std::map<std::string, cv::Mat>& views,
                       const std::string& outputDir) {
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

    // 1. 對每張圖做 IPM
    cv::Mat f = applyIPM(views.at("front"));
    cv::Mat b = applyIPM(views.at("back"));
    cv::Mat l = applyIPM(views.at("left"));
    cv::Mat r = applyIPM(views.at("right"));

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
    int blendW = std::min(W, H) / 4;  // blending 寬度
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

        // 如果目標區域是空的（深灰），直接貼
        cv::Scalar regionMean = cv::mean(region);
        if (regionMean[0] < 50 && regionMean[1] < 50 && regionMean[2] < 50) {
            view.copyTo(region);
        } else {
            // 加權混合
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

    // 8. 中間畫車體
    cv::rectangle(canvas,
        cv::Point(W, H), cv::Point(W * 2, H * 2),
        cv::Scalar(60, 60, 60), -1);
    cv::rectangle(canvas,
        cv::Point(W, H), cv::Point(W * 2, H * 2),
        cv::Scalar(120, 120, 120), 2);
    cv::putText(canvas, "CAR",
        cv::Point(W + W / 2 - 40, H + H / 2 + 10),
        cv::FONT_HERSHEY_SIMPLEX, 1.5,
        cv::Scalar(180, 180, 180), 3);

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
    cv::rectangle(canvasNB, {W, H}, {W * 2, H * 2}, {60, 60, 60}, -1);
    cv::putText(canvasNB, "CAR", {W + W/2 - 40, H + H/2 + 10},
        cv::FONT_HERSHEY_SIMPLEX, 1.5, {180, 180, 180}, 3);
    cv::imwrite(outputDir + "/birdview_no_blend.jpg", canvasNB);
    std::cout << "[stitch] 無 blending 對比圖已儲存\n";

    return canvas;
}
