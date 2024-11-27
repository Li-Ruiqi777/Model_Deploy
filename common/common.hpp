#pragma once
#include "NvInfer.h"
#include "opencv2/opencv.hpp"

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims &dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType &dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

struct Binding
{
    size_t size = 1;  // 数据大小(总字节数)
    size_t dsize = 1; // 数据精度(几字节)
    nvinfer1::Dims dims;
    std::string name;
};

struct Object
{
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0;
};

struct PreParam
{
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};

struct infer_context
{
    infer_context(nvinfer1::ICudaEngine *engine, const cv::Mat &img)
        : num_inputs(0), num_outputs(0)
    {
        cudaStreamCreate(&this->stream);
        this->execution_context = engine->createExecutionContext();
        auto num_bindings = engine->getNbBindings();
        for (int i = 0; i < num_bindings; ++i)
        {
            Binding binding;
            nvinfer1::Dims dims;

            nvinfer1::DataType dtype = engine->getBindingDataType(i);
            std::string name = engine->getBindingName(i);

            binding.name = name;
            binding.dsize = type_to_size(dtype);

            bool IsInput = engine->bindingIsInput(i);

            if (IsInput)
            {
                this->num_inputs += 1;

                // 修改推理上下文中输入的Dim
                dims = nvinfer1::Dims4(1, 3, img.rows, img.cols);
                this->execution_context->setBindingDimensions(i, dims);

                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->input_bindings.push_back(binding);
            }
            else
            {
                this->num_outputs += 1;
                dims = this->execution_context->getBindingDimensions(i);
                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->output_bindings.push_back(binding);
            }
        }
    }
    ~infer_context()
    {
        cudaStreamDestroy(this->stream);
        for (auto &ptr : this->buffers)
        {
            CHECK(cudaFree(ptr));
        }
        this->execution_context->destroy();
    }

    infer_context(const infer_context &) = delete;
    infer_context &operator=(const infer_context &) = delete;

    int num_inputs;
    int num_outputs;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> buffers;
    nvinfer1::IExecutionContext* execution_context;
    cudaStream_t stream;
};
