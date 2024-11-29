#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <vector>

void preprocess(const std::vector<cv::Mat>& frames, float* gpuInput, int batchSize, int channels, int height, int width, cudaStream_t stream);

#endif