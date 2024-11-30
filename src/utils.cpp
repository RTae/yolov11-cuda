#include "utils.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>

std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string &enginePath, nvinfer1::ILogger &logger)
{
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Error loading engine file: " << enginePath << std::endl;
        return nullptr;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "Failed to create TensorRT runtime!" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fileSize));
}

// Function to split a string by a delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}