#ifndef YOLOINFERENCE_H
#define YOLOINFERENCE_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <memory>

std::vector<std::vector<std::vector<float>>> yolov11BatchInference(
    const std::vector<cv::Mat>& frames, 
    nvinfer1::IExecutionContext* context, 
    const nvinfer1::ICudaEngine* engine, 
    void* buffers[], 
    cudaStream_t stream, 
    float confidenceThreshold
);

#endif