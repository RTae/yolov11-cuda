#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h>

void preprocess(const std::vector<cv::Mat> &frames, float *gpuInput, int batchSize, int channels, int height, int width, cudaStream_t stream);

#endif // PREPROCESSOR_H