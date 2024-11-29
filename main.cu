#include "logger.h"
#include "common.h"
#include "yoloInference.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sstream>
#include <map>

// Instantiate a global logger
Logger gLogger;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_path> [--engine_path=PATH] [--batch_size=N] [--confidence_threshold=FLOAT]" << std::endl;
        return -1;
    }

    // Parse arguments
    std::string inputPath;
    std::string enginePath = "./asset/model-weigth/yolo11s.engine";
    int batchSize = 8;
    float confidenceThreshold = 0.7;

    // Parse required input path
    inputPath = argv[1];

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--engine_path=") == 0) {
            enginePath = arg.substr(14); // Extract value after '='
        } else if (arg.find("--batch_size=") == 0) {
            batchSize = std::stoi(arg.substr(13));
        } else if (arg.find("--confidence_threshold=") == 0) {
            confidenceThreshold = std::stof(arg.substr(23));
        }
    }

    // Validate inputs
    if (batchSize <= 0) {
        std::cerr << "Invalid batch size. It must be greater than 0." << std::endl;
        return -1;
    }
    if (confidenceThreshold <= 0.0f || confidenceThreshold > 1.0f) {
        std::cerr << "Invalid confidence threshold. It must be between 0 and 1." << std::endl;
        return -1;
    }

    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    Logger gLogger;
    auto engine = loadEngine(enginePath, gLogger);
    if (!engine) {
        std::cerr << "Error loading TensorRT engine from path: " << enginePath << std::endl;
        return -1;
    }

    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return -1;
    }

    const int inputSize = batchSize * 3 * 640 * 640;
    const int outputSize = batchSize * (80 + 4) * 8400;
    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[1], outputSize * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

if (isVideo) {
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file!" << std::endl;
        return -1;
    }

    // Set up output video writer
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    std::string outputPath = "out_" + inputPath.substr(inputPath.find_last_of("/") + 1);

    // Use H264 codec for MP4 output
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<cv::Mat> batchFrames;

    while (cap.read(frame)) {
        batchFrames.push_back(frame.clone());

        // Process frames in batches
        if (batchFrames.size() == batchSize || cap.get(cv::CAP_PROP_POS_FRAMES) == cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            auto allDetections = yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);

            for (size_t i = 0; i < batchFrames.size(); ++i) {
                if (!allDetections[i].empty()) {
                    drawDetections(batchFrames[i], allDetections[i], classLabels, 640, 640);
                }
                writer.write(batchFrames[i]);
            }
            batchFrames.clear();
        }
    }

    cap.release();
    writer.release();
    } else {
        std::vector<cv::Mat> images;
        cv::Mat img = cv::imread(inputPath);
        if (img.empty()) {
            std::cerr << "Error loading image: " << inputPath << std::endl;
            return -1;
        }
        images.push_back(img);

        // Run inference and get detections
        auto allDetections = yolov11BatchInference(images, context, engine.get(), buffers, stream, confidenceThreshold);

        if (!allDetections.empty() && !allDetections[0].empty()) {
            const auto& detections = allDetections[0]; // Access detections for the first image

            // Draw detections on the image
            drawDetections(img, allDetections[0], classLabels, 640, 640); 

            // Save the image with detections
            std::string outputPath = "out_" + inputPath.substr(inputPath.find_last_of("/") + 1);
            cv::imwrite(outputPath, img);
            std::cout << "Saved output to " << outputPath << std::endl;
        }
    }

    // Destroy the execution context explicitly
    if (context) {
        delete context;
        context = nullptr;
    }

    // Free GPU resources
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));

    return 0;
}
