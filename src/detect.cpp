#include "detect.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <algorithm>

static const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

static cv::Scalar getColor(int id) {
    static std::vector<cv::Scalar> colors;
    if (colors.empty()) {
        cv::RNG rng(42);
        for (int i = 0; i < 80; i++)
            colors.emplace_back(rng.uniform(50, 255),
                                rng.uniform(50, 255),
                                rng.uniform(50, 255));
    }
    return colors[id % 80];
}

cv::Mat detectObjects(const cv::Mat& input,
                      const std::string& modelPath,
                      float confThr, float iouThr) {
    cv::Mat img = input.clone();

    // 1. 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelPath.c_str(), opts);
    std::cout << "[detect] 模型載入成功: " << modelPath << "\n";

    // 2. 前處理
    int origH = img.rows, origW = img.cols;
    const int SZ = 640;

    cv::Mat resized;
    cv::resize(img, resized, {SZ, SZ});
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0,
        {SZ, SZ}, cv::Scalar(), true, false, CV_32F);

    // 3. 建立 input tensor
    std::vector<int64_t> inputShape = {1, 3, SZ, SZ};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, (float*)blob.data,
        blob.total(), inputShape.data(), inputShape.size());

    // 4. 推論
    auto t0 = cv::getTickCount();
    const char* inputNames[]  = {"images"};
    const char* outputNames[] = {"output0"};
    auto results = session.Run(Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1, outputNames, 1);
    double ms = (cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;
    std::cout << "[detect] 推論時間: " << ms << " ms\n";

    // 5. 解析輸出 (1, 84, 8400)
    float* data = results[0].GetTensorMutableData<float>();
    auto shape = results[0].GetTensorTypeAndShapeInfo().GetShape();
    int numCls  = (int)shape[1] - 4;
    int numAnch = (int)shape[2];

    float xRatio = (float)origW / SZ;
    float yRatio = (float)origH / SZ;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> clsIds;

    for (int a = 0; a < numAnch; a++) {
        float cx = data[0 * numAnch + a];
        float cy = data[1 * numAnch + a];
        float bw = data[2 * numAnch + a];
        float bh = data[3 * numAnch + a];

        float maxScore = 0;
        int maxCls = 0;
        for (int c = 0; c < numCls; c++) {
            float s = data[(4 + c) * numAnch + a];
            if (s > maxScore) { maxScore = s; maxCls = c; }
        }
        if (maxScore < confThr) continue;

        int x = (int)((cx - bw / 2) * xRatio);
        int y = (int)((cy - bh / 2) * yRatio);
        int w = (int)(bw * xRatio);
        int h = (int)(bh * yRatio);

        boxes.emplace_back(x, y, w, h);
        confs.push_back(maxScore);
        clsIds.push_back(maxCls);
    }

    // 6. NMS
    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, confs, confThr, iouThr, idx);
    std::cout << "[detect] 偵測到 " << idx.size() << " 個物件\n";

    // 7. 畫框
    for (int i : idx) {
        auto& box = boxes[i];
        auto col = getColor(clsIds[i]);
        cv::rectangle(img, box, col, 2);

        std::string lbl = CLASS_NAMES[clsIds[i]] + " " +
                          std::to_string((int)(confs[i] * 100)) + "%";
        int base;
        auto ts = cv::getTextSize(lbl,
            cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
        cv::rectangle(img,
            {box.x, box.y - ts.height - 6},
            {box.x + ts.width, box.y}, col, -1);
        cv::putText(img, lbl, {box.x, box.y - 4},
            cv::FONT_HERSHEY_SIMPLEX, 0.55,
            cv::Scalar(255, 255, 255), 1);
    }

    return img;
}
