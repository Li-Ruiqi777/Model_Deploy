#include "zero-dce.h"
#include "filesystem.hpp"

ZeroDCE::ZeroDCE(const std::string &engine_file_path)
{
    cudaSetDevice(0);
    std::ifstream file(engine_file_path, std::ios::binary);

    size_t size = 0;
    char *trtModelStream = nullptr;
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    initLibNvInferPlugins(&this->gLogger, "");
    // 创建推理引擎、上下文、流
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
}

ZeroDCE::~ZeroDCE()
{
    this->engine->destroy();
    this->runtime->destroy();
}

cv::Mat ZeroDCE::forward(const cv::Mat &src)
{
    infer_context context(this->engine, src);
    // 分配显存
    for (auto &bindings : context.input_bindings)
    {
        void *ptr;
        CHECK(cudaMallocAsync(&ptr, bindings.size * bindings.dsize, context.stream));
        context.buffers.push_back(ptr);
    }

    for (auto &bindings : context.output_bindings)
    {
        void *ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&ptr, size, context.stream));
        context.buffers.push_back(ptr);
    }

    // 准备输入
    cv::Mat nchw = cv::dnn::blobFromImage(src, 1.0 / 255.0);
    CHECK(cudaMemcpyAsync(
        context.buffers[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, context.stream));
    
    context.execution_context->enqueueV2(context.buffers.data(), context.stream, nullptr);
    
    float *output_buffer = new float[nchw.total() * nchw.elemSize()];
    size_t osize = context.output_bindings[1].size * context.output_bindings[1].dsize;
    CHECK(cudaMemcpyAsync(
        output_buffer, context.buffers[2], osize, cudaMemcpyDeviceToHost, context.stream));
    cudaStreamSynchronize(context.stream);

    // 准备输出
    int channel = context.output_bindings[1].dims.d[1];
    int height = context.output_bindings[1].dims.d[2];
    int width = context.output_bindings[1].dims.d[3];
    cv::Mat result(height, width, CV_32FC3);

    for (int c = 0; c < channel; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                result.at<cv::Vec3f>(h, w)[c] = static_cast<float *>(output_buffer)[c * height * width + h * width + w];
            }
        }
    }
    cv::cvtColor(result, result, cv::COLOR_RGB2GRAY);
    result.convertTo(result, CV_8U, 255.0);
    return result;
}