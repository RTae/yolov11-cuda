#include "logger.h"
#include "common.h"
#include "yoloInference.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

// Instantiate a global logger
Logger gLogger;

// Function to process a single file (image or video)
void processFile(const std::string& inputPath, const std::string& enginePath, int batchSize, float confidenceThreshold) {
    auto engine = loadEngine(enginePath, gLogger);
    if (!engine) {
        std::cerr << "Error loading TensorRT engine from path: " << enginePath << std::endl;
        return;
    }

    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return;
    }

    const int inputSize = batchSize * 3 * 640 * 640;
    const int outputSize = batchSize * (80 + 4) * 8400;
    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[1], outputSize * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    if (isVideo) {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file: " << inputPath << std::endl;
            return;
        }

        std::string outputVideoPath = "out_" + inputPath.substr(inputPath.find_last_of("/") + 1);
        cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(cv::CAP_PROP_FPS),
                               cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat frame;
        std::vector<cv::Mat> batchFrames;
        while (cap.read(frame)) {
            batchFrames.push_back(frame.clone());

            if (batchFrames.size() == batchSize) {
                auto allDetections = yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
                for (size_t i = 0; i < batchFrames.size(); ++i) {
                    drawDetections(batchFrames[i], allDetections[i], classLabels, 640, 640);
                    writer.write(batchFrames[i]);
                }
                batchFrames.clear();
            }
        }

        if (!batchFrames.empty()) {
            auto allDetections = yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
            for (size_t i = 0; i < batchFrames.size(); ++i) {
                drawDetections(batchFrames[i], allDetections[i], classLabels, 640, 640);
                writer.write(batchFrames[i]);
            }
        }

        cap.release();
        writer.release();
        std::cout << "Processed video saved to: " << outputVideoPath << std::endl;
    } else {
        cv::Mat img = cv::imread(inputPath);
        if (img.empty()) {
            std::cerr << "Error loading image: " << inputPath << std::endl;
            return;
        }

        std::vector<cv::Mat> images = {img};
        auto allDetections = yolov11BatchInference(images, context, engine.get(), buffers, stream, confidenceThreshold);
        drawDetections(img, allDetections[0], classLabels, 640, 640);

        std::string outputImagePath = "out_" + inputPath.substr(inputPath.find_last_of("/") + 1);
        cv::imwrite(outputImagePath, img);
        std::cout << "Processed image saved to: " << outputImagePath << std::endl;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));
    delete context;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_paths> [--engine_path=<engine_path>] [--batch_size=<batch_size>] [--confidence_threshold=<confidence_threshold>]" << std::endl;
        return -1;
    }

    std::string inputPaths = argv[1];
    std::string enginePath = "./asset/model-weigth/yolo11s.engine";
    int batchSize = 8;
    float confidenceThreshold = 0.7;

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--engine_path=") == 0) {
            enginePath = arg.substr(arg.find("=") + 1);
        } else if (arg.find("--batch_size=") == 0) {
            batchSize = std::stoi(arg.substr(arg.find("=") + 1));
        } else if (arg.find("--confidence_threshold=") == 0) {
            confidenceThreshold = std::stof(arg.substr(arg.find("=") + 1));
        }
    }

    // Split input paths by comma
    std::vector<std::string> files = split(inputPaths, ',');

    // Create a thread for each file
    std::vector<std::thread> threads;
    for (const auto& file : files) {
        threads.emplace_back(processFile, file, enginePath, batchSize, confidenceThreshold);
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
