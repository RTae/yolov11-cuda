#include "logger.h"

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity != nvinfer1::ILogger::Severity::kINFO) {
        std::cerr << "TensorRT [" << getSeverityString(severity) << "]: " << msg << std::endl;
    }
}

const char* Logger::getSeverityString(Severity severity) const {
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case nvinfer1::ILogger::Severity::kERROR: return "ERROR";
        case nvinfer1::ILogger::Severity::kWARNING: return "WARNING";
        case nvinfer1::ILogger::Severity::kINFO: return "INFO";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
    }
}
