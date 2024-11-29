#include "logger.h"
#include "yoloInference.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

// Instantiate a global logger
Logger gLogger;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    std::string enginePath = "../model-weigth/yolo11s.engine";
    auto engine = loadEngine(enginePath, gLogger);
    if (!engine) {
        std::cerr << "Error loading TensorRT engine!" << std::endl;
        return -1;
    }

    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return -1;
    }

    const int batchSize = 8;
    const int inputSize = batchSize * 3 * 640 * 640;
    const int outputSize = batchSize * (80 + 4) * 8400;
    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[1], outputSize * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float confidenceThreshold = 0.90;

    // Example class labels (COCO dataset)
    std::vector<std::string> classLabels = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                                            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
                                            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                                            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                                            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                                            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                                            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                            "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
                                            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                                            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                                            "toothbrush"};

    if (isVideo) {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file!" << std::endl;
            return -1;
        }

        cv::Mat frame;
        std::vector<cv::Mat> batchFrames;
        while (cap.read(frame)) {
            batchFrames.push_back(frame.clone());

            if (batchFrames.size() == batchSize) {
                auto allDetections = yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
                batchFrames.clear();
            }
        }

        if (!batchFrames.empty()) {
            auto allDetections = yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
        }

        cap.release();
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

        // Draw detections on the image
        if (!allDetections.empty()) {
            drawDetections(img, allDetections[0], classLabels);
        }

        // Save the image with detections
        std::string outputPath = "out_" + inputPath.substr(inputPath.find_last_of("/") + 1);
        cv::imwrite(outputPath, img);
        std::cout << "Saved output to " << outputPath << std::endl;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));

    return 0;
}
