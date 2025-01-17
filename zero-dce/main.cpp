#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include "common.hpp"

#include "zero-dce.h"

void test_Augment_one_image()
{
    const std::string engine_file_path = "E:/DeepLearning/Zero-DCE-improved/zero-dce-fixed.engine";
    const std::string image_path = "E:/Works/WireRopeDocs/experiment/HikCamear-RGB/exposure-time/50us/Image_7.bmp";
    cv::Mat src = cv::imread(image_path);
    cv::resize(src, src, cv::Size(1024, 1024));

    ZeroDCE *zeroDCE = new ZeroDCE(engine_file_path);
    zeroDCE->make_pipe(false);
    zeroDCE->copy_from_Mat(src);
    zeroDCE->infer();
    auto res = zeroDCE->postprocess();
    cv::imshow("res", res);
    cv::waitKey(0);
}

void test_Augment_images()
{
    const std::string engine_file_path = "E:/DeepLearning/Zero-DCE-improved/zero-dce-fixed.engine";
    const std::string image_folder_path = "E:/DeepLearning/0_DataSets/006-rope/004-rope1+2-augmented/COCO_Format/images";

    ZeroDCE *zeroDCE = new ZeroDCE(engine_file_path);
    zeroDCE->make_pipe(false);

    for (const auto &entry : std::filesystem::directory_iterator(image_folder_path))
    {
        if (entry.is_regular_file() && (entry.path().extension() == ".bmp" || entry.path().extension() == ".jpg" || entry.path().extension() == ".png"))
        {
            const std::string image_path = entry.path().string();
            std::cout << "Processing: " << image_path << std::endl;

            cv::Mat src = cv::imread(image_path);
            if (src.empty())
            {
                std::cerr << "Failed to read image: " << image_path << std::endl;
                continue;
            }

            cv::resize(src, src, cv::Size(1024, 1024));
            zeroDCE->copy_from_Mat(src);
            zeroDCE->infer();
            auto res = zeroDCE->postprocess();

            if (!cv::imwrite(image_path, res))
            {
                std::cerr << "Failed to save image: " << image_path << std::endl;
            }
        }
    }

    delete zeroDCE;
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    test_Augment_one_image();
    // test_Augment_images();
}