#include "detect_service.h"

#include "sahi.h"

const std::string MODEL_PATH = "E:/DeepLearning/Model_Deploy/model/yolo11n-rope.engine";
const std::string IMG_PATH = "E:/DeepLearning/Model_Deploy/3.png";
const std::vector<std::string> CLASS_NAMES = {
    "broken",
    "warp",
    "scatter",
    "rust",
    "wear",
};
const std::vector<std::vector<unsigned int>> COLORS = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {255, 255, 0},
};

void test_Detect()
{
    DetectService detect_service(true);
    cv::Mat image = cv::imread(IMG_PATH);
    std::vector<Object> bboxes;
    auto dst = detect_service.predict(image, bboxes);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void test_Slice(int slice_height, int slice_width,
                float overlap_height_ratio, float overlap_width_ratio)
{
    cv::Mat img(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(img.rows, img.cols);

    // Draw the slice regions
    for (const auto &region : sliceRegions)
    {
        Object box{region.first, 0, 0};
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        cv::rectangle(img, box.rect, color, 2);
    }

    cv::imshow("Slice", img);
    cv::waitKey(0);
}

void test_Slice2(int slice_height, int slice_width,
                 float overlap_height_ratio, float overlap_width_ratio)
{
    cv::Mat img = cv::imread(IMG_PATH);
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(img.rows, img.cols,
                                                                                    350, 0, 900, 1024);

    // Draw the slice regions
    for (const auto &region : sliceRegions)
    {
        Object box{region.first, 0, 0};
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        cv::rectangle(img, box.rect, color, 2);
    }

    cv::imshow("Slice", img);
    cv::waitKey(0);
}

void test_SAHI(int slice_height, int slice_width,
               float overlap_height_ratio, float overlap_width_ratio)
{
    cudaSetDevice(0);
    auto model = std::make_unique<YOLOv8>(MODEL_PATH);
    model->make_pipe(false);

    cv::Mat img = cv::imread(IMG_PATH);
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(img.rows, img.cols,
                                                                                    350, 0, 900, 1024);

    std::vector<Object> allObjects;

    int infer_times = 0;
    // 切片并推理
    for (const auto &region : sliceRegions)
    {
        ++infer_times;

        cv::Mat slice = img(region.first);
        std::vector<Object> objects;
        model->copy_from_Mat(slice);
        model->infer();
        model->process_output(objects, 0.1, 5);
        for (auto &object : objects)
        {
            object = SAHI::mapToOriginal(object, region.first);
            allObjects.push_back(object);
        }
    }
    std::cout << "infer_times: " << infer_times << std::endl;

    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto &obj : allObjects)
    {
        boxes.push_back(obj.rect);
        scores.push_back(obj.prob);
    }
    cv::dnn::NMSBoxes(boxes, scores, 0.1f, 0.1f, indices);

    std::vector<Object> filteredObjects;
    for (int &i : indices)
    {
        filteredObjects.push_back(allObjects[i]);
    }

    cv::Mat dst;
    model->draw_objects(img, dst, filteredObjects, CLASS_NAMES, COLORS);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    // test_Detect();
    // test_Slice(128, 128, 0, 0);
    test_Slice2(256, 256, 0, 0);
    test_SAHI(256, 256, 0, 0);
}
