#pragma once
#include "common.hpp"

class SAHI
{
public:
    SAHI(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio);

    // 在图像中计算切片区域
    std::vector<std::pair<cv::Rect, int>> calculateSliceRegions(int image_height, int image_width);

    // 在图像中计算切片区域, 并指定起始和结束坐标,通过(end_x - start_x)计算切片数量
    std::vector<std::pair<cv::Rect, int>> calculateSliceRegions(int image_height, int image_width,
                                                                int start_x, int start_y,
                                                                int end_x, int end_y);

    static Object mapToOriginal(const Object &boundingBox, const cv::Rect &sliceRegion);

private:
    int slice_height_, slice_width_;
    int image_height_, image_width_;
    float overlap_height_ratio_;
    float overlap_width_ratio_;
};