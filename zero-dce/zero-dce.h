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

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat &image);

    void infer();
    cv::Mat postprocess();

    int num_bindings = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

private:
    const cv::Size input_size = cv::Size(1024, 1024); // WH
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};
