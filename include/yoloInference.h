#ifndef YOLO_INFERENCE_H
#define YOLO_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include "postprocessor.h"

// Declare the yolov11BatchInference function
std::vector<std::vector<Detection>> yolov11BatchInference(
    const std::vector<cv::Mat>& frames,
    nvinfer1::IExecutionContext* context,
    const nvinfer1::ICudaEngine* engine,
    void* buffers[],
    cudaStream_t stream,
    float confidenceThreshold);

#endif // YOLO_INFERENCE_H
