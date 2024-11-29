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
    const int inputChannels = 3;
    const int inputHeight = 640;
    const int inputWidth = 640;
    const int outputClasses = 80; // Number of classes
    const int outputBoxes = 8400; // Total number of anchors

    // Preprocess frames into batch
    preprocess(frames, static_cast<float*>(buffers[0]), batchSize, inputChannels, inputHeight, inputWidth, stream);

    // Execute inference using enqueueV3
    context->setInputTensorAddress("images", buffers[0]);   // Set the input tensor address
    context->setTensorAddress("output0", buffers[1]);       // Set the output tensor address

    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to execute inference with TensorRT.");
    }

    // Copy inference results back to host
    const int outputSize = batchSize * (outputClasses + 5) * outputBoxes; // +5 for bbox attributes
    std::vector<float> hostOutput(outputSize);
    CHECK_CUDA(cudaMemcpyAsync(hostOutput.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Log output dimensions
    std::cout << "Model Output Dimensions: Batch Size: " << batchSize
              << ", Output Classes: " << outputClasses
              << ", Output Boxes: " << outputBoxes << std::endl;

    // Postprocess detections for each frame in the batch
    std::vector<std::vector<Detection>> allDetections(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        allDetections[i] = postprocess(
            hostOutput.data() + i * (outputClasses + 5) * outputBoxes, // Offset for each batch
            outputBoxes,           // numDetections (total anchors)
            outputClasses,         // numClasses
            confidenceThreshold,   // confThreshold
            0.45                   // nmsThreshold
        );
    }

    return allDetections;
}
