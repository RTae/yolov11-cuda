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
std::vector<std::vector<std::vector<float>>> yolov11BatchInference(
    const std::vector<cv::Mat>& frames, 
    nvinfer1::IExecutionContext* context, 
    const nvinfer1::ICudaEngine* engine, 
    void* buffers[], 
    cudaStream_t stream, 
    float confidenceThreshold
) {
    const int batchSize = frames.size();
    const int inputChannels = 3;
    const int inputHeight = 640;
    const int inputWidth = 640;
    const int outputClasses = 80; // Number of classes
    const int outputBoxes = 8400; // Total number of anchors

    // Preprocess frames into batch
    preprocess(frames, static_cast<float*>(buffers[0]), batchSize, inputChannels, inputHeight, inputWidth, stream);

    // Execute inference
    context->executeV2(buffers);

    // Copy inference results back to host
    const int outputSize = batchSize * (outputClasses + 4) * outputBoxes;
    std::vector<float> hostOutput(outputSize);
    CHECK_CUDA(cudaMemcpy(hostOutput.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(stream);

    // Postprocess detections for each frame in the batch
    std::vector<std::vector<std::vector<float>>> allDetections(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        allDetections[i] = postprocess(
            hostOutput.data() + i * (outputClasses + 4) * outputBoxes, 
            1, 
            outputClasses, 
            outputBoxes, 
            confidenceThreshold, 
            0.45
        );
    }

    return allDetections;
}

void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, const std::vector<std::string>& classLabels) {
    for (const auto& det : detections) {
        int x = static_cast<int>(det[0]);
        int y = static_cast<int>(det[1]);
        int width = static_cast<int>(det[2]);
        int height = static_cast<int>(det[3]);
        float confidence = det[4];
        int classId = static_cast<int>(det[5]);

        // Draw the bounding box
        cv::rectangle(image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);

        // Prepare label text
        std::string label = classLabels[classId] + " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";

        // Draw the label background
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image, cv::Point(x, y - labelSize.height - 10), cv::Point(x + labelSize.width, y), cv::Scalar(0, 255, 0), cv::FILLED);

        // Put the label text
        cv::putText(image, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    std::string enginePath = "../model-weigth/yolo11s.engine";
    auto engine = loadEngine(enginePath);
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
