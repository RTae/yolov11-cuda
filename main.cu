#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) { // Suppress INFO-level messages
            std::cerr << "TensorRT [" << getSeverityString(severity) << "]: " << msg << std::endl;
        }
    }

private:
    const char* getSeverityString(Severity severity) const {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR: return "ERROR";
            case Severity::kWARNING: return "WARNING";
            case Severity::kINFO: return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
            default: return "UNKNOWN";
        }
    }
};

// Instantiate a global logger
Logger gLogger;

// Load TensorRT engine
std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& enginePath) {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error loading engine file: " << enginePath << std::endl;
        return nullptr;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    auto runtime = nvinfer1::createInferRuntime(gLogger);
    return std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fileSize));
}

// Preprocess input images
void preprocess(const std::vector<cv::Mat>& frames, float* gpuInput, int batchSize, int channels, int height, int width, cudaStream_t stream) {
    std::vector<float> batchInput(batchSize * channels * height * width);

    for (int i = 0; i < batchSize; ++i) {
        cv::Mat resized, floatImage;
        cv::resize(frames[i], resized, cv::Size(width, height));
        resized.convertTo(floatImage, CV_32FC3, 1 / 255.0); // Normalize to [0, 1]
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

        // NHWC to NCHW format
        std::vector<cv::Mat> channelsSplit(channels);
        cv::split(floatImage, channelsSplit);
        for (int c = 0; c < channels; ++c) {
            std::memcpy(batchInput.data() + i * channels * height * width + c * height * width,
                        channelsSplit[c].data, height * width * sizeof(float));
        }
    }

    // Copy to GPU
    CHECK_CUDA(cudaMemcpyAsync(gpuInput, batchInput.data(), batchInput.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}

// Helper function for Non-Maximum Suppression (NMS)
std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iouThreshold) {
    std::vector<int> indices;
    std::vector<int> sortedIndices(scores.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    std::sort(sortedIndices.begin(), sortedIndices.end(), [&scores](int i, int j) {
        return scores[i] > scores[j];
    });

    while (!sortedIndices.empty()) {
        int current = sortedIndices.front();
        indices.push_back(current);
        sortedIndices.erase(sortedIndices.begin());

        auto it = std::remove_if(sortedIndices.begin(), sortedIndices.end(), [&](int idx) {
            float iou = (float)(boxes[current] & boxes[idx]).area() / (float)(boxes[current] | boxes[idx]).area();
            return iou > iouThreshold;
        });
        sortedIndices.erase(it, sortedIndices.end());
    }

    return indices;
}

// Postprocess YOLO output
std::vector<std::vector<float>> postprocess(const float* output, int batchSize, int numClasses, int numAnchors, float confThreshold, float iouThreshold) {
    std::vector<std::vector<float>> detections;

    for (int b = 0; b < batchSize; ++b) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int anchor = 0; anchor < numAnchors; ++anchor) {
            const float* anchorData = output + b * (numClasses + 4) * numAnchors + anchor;

            // Bounding box attributes
            float x = anchorData[0 * numAnchors];
            float y = anchorData[1 * numAnchors];
            float w = anchorData[2 * numAnchors];
            float h = anchorData[3 * numAnchors];

            // Class probabilities directly include objectness
            const float* classScores = anchorData + 4 * numAnchors;
            int maxClassId = std::distance(classScores, std::max_element(classScores, classScores + numClasses));
            float confidence = classScores[maxClassId];

            if (confidence < confThreshold) continue;

            // Convert to top-left corner coordinates
            int x1 = static_cast<int>(x - w / 2.0f);
            int y1 = static_cast<int>(y - h / 2.0f);
            int x2 = static_cast<int>(x + w / 2.0f);
            int y2 = static_cast<int>(y + h / 2.0f);

            boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            confidences.push_back(confidence);
            classIds.push_back(maxClassId);
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> nmsIndices = nms(boxes, confidences, iouThreshold);

        for (int idx : nmsIndices) {
            const auto& box = boxes[idx];
            detections.push_back({
                (float)box.x, (float)box.y, (float)box.width, (float)box.height, confidences[idx], (float)classIds[idx]
            });
        }
    }

    return detections;
}

// YOLOv11 inference
void yolov11BatchInference(std::vector<cv::Mat>& frames, nvinfer1::IExecutionContext* context, const nvinfer1::ICudaEngine* engine, void* buffers[], cudaStream_t stream, float confidenceThreshold) {
    const int batchSize = frames.size();
    const int inputChannels = 3;
    const int inputHeight = 640;
    const int inputWidth = 640;
    const int outputClasses = 80; // Number of classes
    const int outputBoxes = 8400; // Total number of anchors

    preprocess(frames, static_cast<float*>(buffers[0]), batchSize, inputChannels, inputHeight, inputWidth, stream);

    context->executeV2(buffers);

    const int outputSize = batchSize * (outputClasses + 4) * outputBoxes;
    std::vector<float> hostOutput(outputSize);
    CHECK_CUDA(cudaMemcpy(hostOutput.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    auto detections = postprocess(hostOutput.data(), batchSize, outputClasses, outputBoxes, confidenceThreshold, 0.45);

    for (const auto& det : detections) {
        std::cout << "BBox: [" << det[0] << ", " << det[1] << ", " << det[2] << ", " << det[3]
                  << "], Confidence: " << det[4] << ", Class: " << det[5] << std::endl;
    }
}

bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    bool isVideo = endsWith(inputPath, ".mp4");

    std::string enginePath = "../model-weigth/yolo11s.engine";
    auto engine = loadEngine(enginePath);
    if (!engine) return -1;

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

    float confidenceThreshold = 0.9;

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
                yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
                batchFrames.clear();
            }
        }

        if (!batchFrames.empty()) {
            yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
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

        yolov11BatchInference(images, context, engine.get(), buffers, stream, confidenceThreshold);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));

    return 0;
}