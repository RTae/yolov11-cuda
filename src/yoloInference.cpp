#include "yoloInference.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "utils.h"

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
