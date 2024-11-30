#include "preprocessor.h"
#include "utils.h"

// CUDA kernel for preprocessing
__global__ void preprocessKernel(
    const unsigned char *input,
    float *output,
    int inputWidth,
    int inputHeight,
    int resizedWidth,
    int resizedHeight,
    int channels,
    int batchIndex,
    int batchStride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < resizedWidth && y < resizedHeight)
    {
        for (int c = 0; c < channels; ++c)
        {
            // Resize coordinates
            int srcX = x * inputWidth / resizedWidth;
            int srcY = y * inputHeight / resizedHeight;

            // Input index (NHWC format)
            int inputIdx = (srcY * inputWidth + srcX) * channels + c;

            // Output index (NCHW format)
            int outputIdx = batchIndex * batchStride + c * resizedHeight * resizedWidth + y * resizedWidth + x;

            // Normalize to [0, 1]
            output[outputIdx] = input[inputIdx] / 255.0f;
        }
    }
}

void preprocess(const std::vector<cv::Mat> &frames, float *gpuInput, int batchSize, int channels, int height, int width, cudaStream_t stream)
{
    const int batchStride = channels * height * width;

    for (int i = 0; i < batchSize; ++i)
    {
        const cv::Mat &frame = frames[i];
        unsigned char *d_input;
        size_t inputSize = frame.total() * frame.elemSize();

        // Allocate device memory for the input frame
        CHECK_CUDA(cudaMalloc(&d_input, inputSize));

        // Copy input frame to GPU
        CHECK_CUDA(cudaMemcpyAsync(d_input, frame.data, inputSize, cudaMemcpyHostToDevice, stream));

        // Define CUDA grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Launch CUDA kernel
        preprocessKernel<<<gridDim, blockDim, 0, stream>>>(
            d_input, gpuInput, frame.cols, frame.rows, width, height, channels, i, batchStride);

        // Free device memory for the input frame
        CHECK_CUDA(cudaFree(d_input));
    }
}
