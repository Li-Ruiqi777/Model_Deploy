#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include "common.hpp"

#include "zero-dce.h"

int main()
{
    const std::string engine_file_path = "E:/DeepLearning/Zero-DCE-improved/zero-dce.engine";
    const std::string image_path = "E:/DeepLearning/Zero-DCE-improved/src/data/test_data/rope/Image_17.bmp";
    cv::Mat src = cv::imread(image_path);
    cv::resize(src, src, cv::Size(2048, 1024));

    ZeroDCE *zeroDCE = new ZeroDCE(engine_file_path);
    zeroDCE->make_pipe(true);
    zeroDCE->copy_from_Mat(src);
    zeroDCE->infer();
    auto res = zeroDCE->postprocess();
    cv::imshow("res", res);
    cv::waitKey(0);

    return 0;

}