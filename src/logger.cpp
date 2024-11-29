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

void printBindingsInfo(const nvinfer1::ICudaEngine* engine) {
    std::cout << "=== TensorRT Bindings ===" << std::endl;

    int nbIOTensors = engine->getNbIOTensors();  // Get the number of I/O tensors
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* tensorName = engine->getIOTensorName(i);  // Get tensor name
        nvinfer1::Dims dims = engine->getTensorShape(tensorName);  // Get tensor dimensions
        bool isInput = engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;

        std::cout << (isInput ? "Input " : "Output ") << i << ": " << tensorName << " | Dimensions: ";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================" << std::endl;
}