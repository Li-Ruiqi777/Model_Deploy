#include "detect_service.h"

std::vector<std::string> DetectService::CLASS_NAMES = {
    "broken",
    "warp",
    "scatter",
    "rust",
    "wear",
};
std::vector<std::vector<unsigned int>> DetectService::COLORS = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {255, 255, 0},
};

DetectService::DetectService(bool useSAHI_) : useSAHI(useSAHI_)
{
    cudaSetDevice(0);
    this->model = std::make_unique<YOLOv8>(this->engine_file_path);
    this->model->make_pipe(false);

    if (this->useSAHI)
        this->sahi = std::make_unique<SAHI>(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
}

DetectService::~DetectService()
{
    this->model.reset();
}

cv::Mat DetectService::predict(const cv::Mat &image, std::vector<Object> &results,
                               const float score_thres, const float iou_thres,
                               const int topk, const int num_labels)
{
    if (this->useSAHI)
        return this->forward_with_sahi(image, results, score_thres, iou_thres, topk, num_labels);
    else
        return this->forward_without_sahi(image, results, score_thres, iou_thres, topk, num_labels);
}

cv::Mat DetectService::forward_with_sahi(const cv::Mat &image, std::vector<Object> &results,
                                         const float score_thres, const float iou_thres,
                                         const int topk, const int num_labels)
{
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi->calculateSliceRegions(image.rows, image.cols);

    std::vector<Object> allObjects;

    int infer_times = 0;
    // 切片并推理
    for (const auto &region : sliceRegions)
    {
        // 只推理x在图片宽度的1/4~3/4的区域
        int center_x = region.first.x + region.first.width / 2;
        if (center_x < image.cols * 0.25 || center_x > image.cols * 0.75)
            continue;

        ++infer_times;

        cv::Mat slice = image(region.first);
        std::vector<Object> objects;
        this->model->copy_from_Mat(slice);
        this->model->infer();
        this->model->process_output(objects, 0.1, 5);
        for (auto &object : objects)
        {
            object = SAHI::mapToOriginal(object, region.first);
            allObjects.push_back(object);
        }
    }
    std::cout << "infer_times: " << infer_times << std::endl;

    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto &obj : allObjects)
    {
        boxes.push_back(obj.rect);
        scores.push_back(obj.prob);
    }
    cv::dnn::NMSBoxes(boxes, scores, 0.1f, 0.1f, indices);

    for (int &i : indices)
    {
        results.push_back(allObjects[i]);
    }

    cv::Mat dst;
    this->model->draw_objects(image, dst, results, CLASS_NAMES, COLORS);
    return dst;
}

cv::Mat DetectService::forward_without_sahi(const cv::Mat &image, std::vector<Object> &results,
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