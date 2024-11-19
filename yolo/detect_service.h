#pragma once
#include "yolov8.h"
#include <string>
#include "opencv2/opencv.hpp"

class DetectService
{
public:
    DetectService();
    ~DetectService();

    /**
     * @brief 进行一次推理
     * @param image 输入图片
     * @param results 预测出的所有bbox
     * @param score_thres 置信度阈值
     * @param iou_thres NMS阈值
     * @param topk 保留k个结果
     * @param num_labels 类别的数量
     * @return 绘制了结果的图片
     */
    cv::Mat predict(const cv::Mat &image, std::vector<Object> &results,
                    const float score_thres = 0.25f, const float iou_thres = 0.65f,
                    const int topk = 10, const int num_labels = 1);

private:
    std::unique_ptr<YOLOv8> model;
    const cv::Size input_size = cv::Size(800, 800);
    const std::vector<std::string> CLASS_NAMES = {
        "work_piece",
    };
    const std::vector<std::vector<unsigned int>> COLORS = {
        {255, 0, 0},
    };
    const std::string engine_file_path = "E:/DeepLearning/yolo-utils/weights/yolov8n.engine";
};
