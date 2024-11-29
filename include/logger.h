#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>  // Include TensorRT API
#include <iostream>   // For std::cerr

class Logger : public nvinfer1::ILogger {
public:
    // Override the log method from nvinfer1::ILogger
    void log(Severity severity, const char* msg) noexcept override;

private:
    const char* getSeverityString(Severity severity) const;  // Helper to stringify severity
};

#endif
