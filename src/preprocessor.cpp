#include "preprocessor.h"
#include "utils.h"

void preprocess(const std::vector<cv::Mat> &frames, float *gpuInput, int batchSize, int channels, int height, int width, cudaStream_t stream)
{
    std::vector<float> batchInput(batchSize * channels * height * width);

    for (int i = 0; i < batchSize; ++i)
    {
        cv::Mat resized, floatImage;
        cv::resize(frames[i], resized, cv::Size(width, height));
        resized.convertTo(floatImage, CV_32FC3, 1 / 255.0); // Normalize to [0, 1]
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

        // NHWC to NCHW format
        std::vector<cv::Mat> channelsSplit(channels);
        cv::split(floatImage, channelsSplit);
        for (int c = 0; c < channels; ++c)
        {
            std::memcpy(batchInput.data() + i * channels * height * width + c * height * width,
                        channelsSplit[c].data, height * width * sizeof(float));
        }
    }

    // Copy to GPU
    CHECK_CUDA(cudaMemcpyAsync(gpuInput, batchInput.data(), batchInput.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}
