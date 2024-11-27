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
    const std::string engine_file_path = "E:/DeepLearning/Zero-DCE-improved/zero-dce_fixed.engine";
    const std::string image_path = "E:/DeepLearning/Zero-DCE-improved/src/data/test_data/rope/Image_17.bmp";

    ZeroDCE *zeroDCE = new ZeroDCE(engine_file_path);

    std::vector<std::thread> tasks;

    cv::Mat src = cv::imread(image_path);
    cv::resize(src, src, cv::Size(1024, 1024));

    std::function<void(int)> f = [=](int i)
    {
        auto res = zeroDCE->forward(src);
        std::string name = std::to_string(i) + ".bmp";
        // cv::imshow("111",res);
        // cv::waitKey(0);
        cv::imwrite(name, res);
    };

    for (int i = 0; i < 12; ++i)
    {
        tasks.emplace_back(f, i);
    }

    for (auto &t : tasks)
    {
        t.join();
    }

    return 0;
}