#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>

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

// Post-processing function to parse YOLOv11 output
void parseBatchDetections(float* output, int batchSize, int outputSizePerFrame, float confidenceThreshold) {
    std::cout << "Batch Detections:" << std::endl;

    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Frame " << b + 1 << " detections:" << std::endl;
        for (int i = 0; i < outputSizePerFrame; i += 7) { // Adjust stride based on YOLOv11's output format
            float confidence = output[b * outputSizePerFrame + i + 4]; // Confidence score
            if (confidence >= confidenceThreshold) {
                int classId = static_cast<int>(output[b * outputSizePerFrame + i + 5]); // Class ID
                float x = output[b * outputSizePerFrame + i + 0]; // Bounding box center x
                float y = output[b * outputSizePerFrame + i + 1]; // Bounding box center y
                float w = output[b * outputSizePerFrame + i + 2]; // Bounding box width
                float h = output[b * outputSizePerFrame + i + 3]; // Bounding box height

                float x1 = x - w / 2.0f;
                float y1 = y - h / 2.0f;

                std::cout << "  Class ID: " << classId
                          << ", Confidence: " << confidence
                          << ", BBox: [" << x1 << ", " << y1
                          << ", " << w << ", " << h << "]" << std::endl;
            }
        }
    }
}

// YOLOv11 inference function for batch
void yolov11BatchInference(std::vector<cv::Mat>& frames, nvinfer1::IExecutionContext* context, const nvinfer1::ICudaEngine* engine, void* buffers[], cudaStream_t stream, float confidenceThreshold) {
    const int batchSize = frames.size();
    const int inputIndex = context->getEngine().getBindingIndex("input");
    const int outputIndex = context->getEngine().getBindingIndex("output");
    const int inputSizePerFrame = 640 * 640 * 3; // H x W x C
    const int outputSizePerFrame = 1000; // Adjust based on YOLOv11's output size

    // Preprocess frames into batch
    std::vector<float> batchInput(batchSize * inputSizePerFrame);
    for (int b = 0; b < batchSize; ++b) {
        cv::Mat resized;
        cv::resize(frames[b], resized, cv::Size(640, 640));
        cv::Mat floatImage;
        resized.convertTo(floatImage, CV_32FC3, 1 / 255.0);
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

        std::memcpy(batchInput.data() + b * inputSizePerFrame, floatImage.data, inputSizePerFrame * sizeof(float));
    }

    // Upload to GPU
    CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], batchInput.data(), batchSize * inputSizePerFrame * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Run inference
    context->enqueueV2(buffers, stream, nullptr);

    // Retrieve results
    const int outputSize = batchSize * outputSizePerFrame;
    std::vector<float> batchOutput(outputSize);
    CHECK_CUDA(cudaMemcpyAsync(batchOutput.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Parse and log results
    parseBatchDetections(batchOutput.data(), batchSize, outputSizePerFrame, confidenceThreshold);
}

int main() {
    // Open video
    std::string videoPath = "../asset/test.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file!" << std::endl;
        return -1;
    }

    // Load TensorRT engine
    std::string enginePath = "../model-weigth/yolo11s.engine";
    auto engine = loadEngine(enginePath);
    if (!engine) return -1;

    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return -1;
    }

    // Allocate buffers
    const int batchSize = 8;
    const int inputSize = batchSize * 640 * 640 * 3;
    const int outputSize = batchSize * 1000;
    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers[1], outputSize * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float confidenceThreshold = 0.5f;

    // Process video in batches
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

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));
    cap.release();

    return 0;
}
