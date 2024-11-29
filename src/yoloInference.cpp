#include "yoloInference.h"
#include "postprocessor.h"
#include "preprocessor.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

std::vector<std::vector<Detection>> yolov11BatchInference(
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

    // Set tensor addresses
    context->setTensorAddress("images", buffers[0]);  // Set input buffer
    context->setTensorAddress("output0", buffers[1]); // Set output buffer

    // Execute inference using enqueueV3
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to execute inference with TensorRT.");
    }

    // Copy inference results back to host
    const int outputSize = batchSize * (outputClasses + 4) * outputBoxes;
    std::vector<float> hostOutput(outputSize);
    CHECK_CUDA(cudaMemcpyAsync(hostOutput.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Postprocess detections for each frame in the batch
    std::vector<std::vector<Detection>> allDetections(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        allDetections[i] = postprocess(
            hostOutput.data() + i * (outputClasses + 4) * outputBoxes, // Offset for each batch
            outputBoxes,           // numDetections (total anchors)
            outputClasses,         // numClasses
            confidenceThreshold,   // confThreshold
            0.45                   // nmsThreshold
        );
    }

    return allDetections;
}
