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
        // Suppress info-level messages to reduce verbosity
        if (severity != Severity::kINFO) {
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
    return std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr));
}

// Post-processing function to parse YOLOv11 output
void parseBatchDetections(float* output, int batchSize, int outputSize, float confidenceThreshold) {
    std::cout << "Batch Detections:" << std::endl;

    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Frame " << b + 1 << " detections:" << std::endl;
        for (int i = 0; i < outputSize; i += 7) { // Adjust stride based on YOLOv11's output format
            float confidence = output[b * outputSize + i + 4]; // Confidence score
            if (confidence >= confidenceThreshold) {
                int classId = static_cast<int>(output[b * outputSize + i + 5]); // Class ID
                float x = output[b * outputSize + i + 0]; // Bounding box center x
                float y = output[b * outputSize + i + 1]; // Bounding box center y
                float w = output[b * outputSize + i + 2]; // Bounding box width
                float h = output[b * outputSize + i + 3]; // Bounding box height

                // Convert center coordinates to top-left corner
                float x1 = x - w / 2.0f;
                float y1 = y - h / 2.0f;

                // Log the result
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
    const int inputIndex = engine->getBindingIndex("input");  // Adjust if different
    const int outputIndex = engine->getBindingIndex("output"); // Adjust if different
    const int inputSizePerFrame = 640 * 640 * 3; // Size of one input frame in float (H x W x C)
    const int outputSizePerFrame = 1000;        // Adjust based on YOLOv11 output size

    // Preprocess all frames into a single batch
    std::vector<float> batchInput(batchSize * inputSizePerFrame);
    for (int b = 0; b < batchSize; ++b) {
        cv::Mat resized;
        cv::resize(frames[b], resized, cv::Size(640, 640)); // Resize to YOLOv11 input size
        cv::Mat floatImage;
        resized.convertTo(floatImage, CV_32FC3, 1 / 255.0);
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

        // Copy frame data into the batch input
        std::memcpy(batchInput.data() + b * inputSizePerFrame, floatImage.data, inputSizePerFrame * sizeof(float));
    }

    // Upload batch input to GPU
    CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], batchInput.data(), batchSize * inputSizePerFrame * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Execute the network
    context->enqueueV2(buffers, stream, nullptr);

    // Retrieve results
    const int outputSize = batchSize * outputSizePerFrame;
    std::vector<float> batchOutput(outputSize);
    CHECK_CUDA(cudaMemcpyAsync(batchOutput.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Parse and log detections for the batch
    parseBatchDetections(batchOutput.data(), batchSize, outputSizePerFrame, confidenceThreshold);
}

int main() {
    // Open the video file
    std::string videoPath = "../asset/test.mp4"; // Replace with your video file path
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file!" << std::endl;
        return -1;
    }

    // Load TensorRT engine for YOLOv11
    std::string enginePath = "../model-weigth/yolo11s.engine"; // Replace with your YOLOv11 TensorRT engine path
    auto engine = loadEngine(enginePath);
    if (!engine) return -1;

    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error creating execution context!" << std::endl;
        return -1;
    }

    // Allocate GPU buffers
    const int batchSize = 8; // Batch size supported by your model
    const int inputSize = batchSize * 640 * 640 * 3; // Input size for the batch
    const int outputSize = batchSize * 1000;         // Output size for the batch
    void* buffers[2];
    CHECK_CUDA(cudaMalloc(&buffers[0], inputSize * sizeof(float))); // Input buffer
    CHECK_CUDA(cudaMalloc(&buffers[1], outputSize * sizeof(float))); // Output buffer

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Confidence threshold for detections
    float confidenceThreshold = 0.5f;

    // Process video in batches
    cv::Mat frame;
    std::vector<cv::Mat> batchFrames;
    while (cap.read(frame)) {
        batchFrames.push_back(frame.clone());

        // If batch is full, process it
        if (batchFrames.size() == batchSize) {
            yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
            batchFrames.clear();
        }
    }

    // Process any remaining frames in the batch
    if (!batchFrames.empty()) {
        yolov11BatchInference(batchFrames, context, engine.get(), buffers, stream, confidenceThreshold);
    }

    // Release resources
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(buffers[0]));
    CHECK_CUDA(cudaFree(buffers[1]));
    cap.release();
    context->destroy();

    return 0;
}
