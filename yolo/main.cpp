#include "detect_service.h"

int main()
{
    DetectService detect_service;
    cv::Mat image = cv::imread("E:/DeepLearning/0_DataSets/005-work_piece/images/train/001.jpg");
    std::vector<Object> bboxes;
    auto dst = detect_service.predict(image, bboxes);
    cv::imshow("dst", image);
    cv::waitKey(0);
}
