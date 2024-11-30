#ifndef UTILS_H
#define UTILS_H

#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

// CUDA error-checking macro
#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// Load a TensorRT engine from a file
std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string &enginePath, nvinfer1::ILogger &logger);

// Split a string by a given delimiter
std::vector<std::string> split(const std::string& str, char delimiter);

#endif
