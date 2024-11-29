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
    float confidenceThreshold) {

    const int batchSize = frames.size();
    const int numClasses = 80;
    const int numAnchors = 8400;

    // Preprocess input frames
    preprocess(frames, static_cast<float*>(buffers[0]), batchSize, 3, 640, 640, stream);

    // Execute inference using enqueueV3
    context->setInputTensorAddress("images", buffers[0]);   // Set the input tensor address
    context->setTensorAddress("output0", buffers[1]);       // Set the output tensor address

    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to execute inference with TensorRT.");
    }

    // Copy inference results back to host
    const int outputSize = batchSize * (numClasses + 4) * numAnchors;
    std::vector<float> hostOutput(outputSize);
    CHECK_CUDA(cudaMemcpyAsync(hostOutput.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Postprocess detections for each frame in the batch
    std::vector<std::vector<Detection>> allDetections(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        allDetections[i] = postprocess(
            hostOutput.data() + i * (numClasses + 4) * numAnchors,
            numAnchors, numClasses, confidenceThreshold, 0.45);
    }

    return allDetections;
}
