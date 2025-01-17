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
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);

    // 获取输入、输出信息
    this->num_bindings = this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i)
    {
        Binding binding;
        nvinfer1::Dims dims;

        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string name = this->engine->getBindingName(i);

        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);

        if (IsInput)
        {
            this->num_inputs += 1;

            // 修改推理上下文中输入的Dim
            // dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX); //获取最优尺寸
            dims = nvinfer1::Dims4(1, 3, input_size.height, input_size.width); // 手动设置输入尺寸
            this->context->setBindingDimensions(i, dims);

            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else
        {
            this->num_outputs += 1;
            dims = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
        }
    }
}

ZeroDCE::~ZeroDCE()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();

    cudaStreamDestroy(this->stream);
    for (auto &ptr : this->device_ptrs)
    {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs)
    {
        CHECK(cudaFreeHost(ptr));
    }
}

// 分配显存及warmup
void ZeroDCE::make_pipe(bool warmup)
{
    for (auto &bindings : this->input_bindings)
    {
        void *d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto &bindings : this->output_bindings)
    {
        void *d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup)
    {
        for (int i = 0; i < 10; i++)
        {
            for (auto &bindings : this->input_bindings)
            {
                size_t size = bindings.size * bindings.dsize;
                void *h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

// 把Mat数据拷贝到显存
void ZeroDCE::copy_from_Mat(const cv::Mat &image)
{
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0 / 255.0);

    auto &in_binding = this->input_bindings[0];
    int height = in_binding.dims.d[2];
    int width = in_binding.dims.d[3];

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
}

void ZeroDCE::infer()
{
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);

    // 拷贝所有输出
    // for (int i = 0; i < this->num_outputs; i++)
    // {
    //     size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
    //     CHECK(cudaMemcpyAsync(
    //         this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    // }

    // 只拷贝图像增强binding的输出(提速)
    size_t osize = this->output_bindings[1].size * this->output_bindings[1].dsize;
    CHECK(cudaMemcpyAsync(
        this->host_ptrs[1], this->device_ptrs[1 + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("inference time: %f ms\n", elapsed.count());
}

// 把nchw的数据拷贝到Mat
cv::Mat ZeroDCE::postprocess()
{
    auto &out_binding = this->output_bindings[1];

    int channel = out_binding.dims.d[1];
    int height = out_binding.dims.d[2];
    int width = out_binding.dims.d[3];
    cv::Mat result(height, width, CV_32FC3);

    for (int c = 0; c < channel; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                result.at<cv::Vec3f>(h, w)[c] = static_cast<float *>(this->host_ptrs[1])[c * height * width + h * width + w];
            }
        }
    }
    // cv::cvtColor(result, result, cv::COLOR_RGB2GRAY);
    result.convertTo(result, CV_8UC1, 255.0);
    return result;
}