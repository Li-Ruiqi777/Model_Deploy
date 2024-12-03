#include "sahi.h"

SAHI::SAHI(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio)
    : slice_height_(slice_height), slice_width_(slice_width),
      overlap_height_ratio_(overlap_height_ratio), overlap_width_ratio_(overlap_width_ratio)
{
}

std::vector<std::pair<cv::Rect, int>> SAHI::calculateSliceRegions(int image_height, int image_width)
{
    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    int y_max = image_height - slice_height_;
    int x_max = image_width - slice_width_;

    // 估计vector的大小以防止内存分配
    int num_rows = (y_max / step_height) + 1;
    int num_cols = (x_max / step_width) + 1;
    std::vector<std::pair<cv::Rect, int>> regions;
    regions.reserve(num_rows * num_cols);

    int index = 0;

    image_width_ = image_width;
    image_height_ = image_height;

    for (int y = 0; y < image_height; y += step_height)
    {
        for (int x = 0; x < image_width; x += step_width)
        {
            int width = slice_width_;
            int height = slice_height_;

            int temp_x = x;
            int temp_y = y;

            // 调整靠近边界的slice的ROI大小,避免切到边界外的地方
            if (x + width > image_width)
                temp_x -= (x + width) - image_width;
            if (y + height > image_height)
                temp_y -= (y + height) - image_height;

            regions.emplace_back(cv::Rect(temp_x, temp_y, width, height), index++);
        }
    }
    return regions;
}

std::vector<std::pair<cv::Rect, int>> SAHI::calculateSliceRegions(int image_height, int image_width,
                                                                  int start_x, int start_y,
                                                                  int end_x, int end_y)
{
    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    // 确保起始点和终止点在图片范围内
    start_x = std::max(start_x, 0);
    start_y = std::max(start_y, 0);
    end_x = std::min(end_x, image_width);
    end_y = std::min(end_y, image_height);

    // 计算切片的最大Y和X坐标
    int y_max = end_y - slice_height_;
    int x_max = end_x - slice_width_;

    // 估计vector的大小以防止内存分配
    int num_rows = (y_max - start_y) / step_height + 1;
    int num_cols = (x_max - start_x) / step_width + 1;
    std::vector<std::pair<cv::Rect, int>> regions;
    regions.reserve(num_rows * num_cols);

    int index = 0;

    image_width_ = image_width;
    image_height_ = image_height;

    for (int y = start_y; y <= y_max; y += step_height)
    {
        for (int x = start_x; x <= x_max; x += step_width)
        {
            int width = slice_width_;
            int height = slice_height_;

            int temp_x = x;
            int temp_y = y;

            // 调整靠近边界的slice的ROI大小,避免切到边界外的地方
            if (x + width > image_width)
                temp_x -= (x + width) - image_width;
            if (y + height > image_height)
                temp_y -= (y + height) - image_height;

            // 创建并存储包含起始坐标和终止坐标信息的slice区域
            regions.emplace_back(cv::Rect(temp_x, temp_y, width, height), index++);
        }
    }
    return regions;
}

Object SAHI::mapToOriginal(const Object &boundingBox, const cv::Rect &sliceRegion)
{
    cv::Rect_<float> newRect = boundingBox.rect;
    newRect.x += sliceRegion.x;
    newRect.y += sliceRegion.y;

    return {newRect, boundingBox.label, boundingBox.prob};
}