
#include "detect_service.h"

DetectService::DetectService()
{
    cudaSetDevice(0);
    this->model = std::make_unique<YOLOv8>(this->engine_file_path);
    this->model->make_pipe(true);
}

DetectService::~DetectService()
{
    this->model.reset();
}

cv::Mat DetectService::predict(const cv::Mat &image, std::vector<Object> &results,
                               const float score_thres, const float iou_thres,
                               const int topk, const int num_labels)
{
    cv::Mat dst = image.clone();
    this->model->copy_from_Mat(dst);
    this->model->infer();
    this->model->postprocess(results, score_thres, iou_thres, topk, num_labels);
    this->model->draw_objects(image, dst, results, CLASS_NAMES, COLORS);
    return dst;
}
