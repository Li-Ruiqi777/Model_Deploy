#pragma once
#include "yolov8.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "sahi.h"

class DetectService
{
public:
    DetectService(bool useSAHI_ = true);
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
                    const int topk = 30, const int num_labels = 5);

    static std::vector<std::string> CLASS_NAMES;
    static std::vector<std::vector<unsigned int>> COLORS;

private:
    cv::Mat forward_with_sahi(const cv::Mat &image, std::vector<Object> &results,
                              const float score_thres, const float iou_thres,
                              const int topk, const int num_labels);

    cv::Mat forward_without_sahi(const cv::Mat &image, std::vector<Object> &results,
                                 const float score_thres, const float iou_thres,
                                 const int topk, const int num_labels);

    std::unique_ptr<YOLOv8> model;
    std::unique_ptr<SAHI> sahi;
    bool useSAHI;
    int slice_height = 512;
    int slice_width = 512;
    float overlap_height_ratio = 0;
    float overlap_width_ratio = 0;

    const cv::Size input_size = cv::Size(1024, 1024);
    const std::string engine_file_path = "E:/DeepLearning/Model_Deploy/model/yolo11n-rope.engine";
};
