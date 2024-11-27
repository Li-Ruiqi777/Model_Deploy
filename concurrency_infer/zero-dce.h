#pragma once
#include "NvInferPlugin.h"
#include "common.hpp"

class ZeroDCE
{
public:
    explicit ZeroDCE(const std::string &engine_file_path);
    ~ZeroDCE();
    ZeroDCE() = delete;
    ZeroDCE(const ZeroDCE &) = delete;
    ZeroDCE &operator=(const ZeroDCE &) = delete;

    cv::Mat forward(const cv::Mat &src);

private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};
