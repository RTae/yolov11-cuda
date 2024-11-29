#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "postprocessor.h"

std::vector<Detection> postprocess(const float* output, int numDetections, int numClasses, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    // Parse detections
    for (int i = 0; i < numDetections; ++i) {
        float objectness = output[i * (numClasses + 5) + 4]; // Objectness score
        if (objectness < confThreshold) continue;

        // Bounding box attributes
        float x = output[i * (numClasses + 5)];
        float y = output[i * (numClasses + 5) + 1];
        float w = output[i * (numClasses + 5) + 2];
        float h = output[i * (numClasses + 5) + 3];

        int x1 = static_cast<int>(x - w / 2.0f);
        int y1 = static_cast<int>(y - h / 2.0f);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        cv::Rect bbox(x1, y1, width, height);

        // Class probabilities
        const float* classScores = output + i * (numClasses + 5) + 5;
        int maxClassId = std::max_element(classScores, classScores + numClasses) - classScores;
        float confidence = objectness * classScores[maxClassId];

        if (confidence >= confThreshold) {
            boxes.push_back(bbox);
            classIds.push_back(maxClassId);
            confidences.push_back(confidence);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Create detections
    for (int idx : indices) {
        detections.push_back(Detection(classIds[idx], confidences[idx], boxes[idx]));
    }

    return detections;
}

void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, const std::vector<std::string>& classLabels) {
    for (const auto& det : detections) {
        // Ensure the detection data has the required fields (x, y, width, height, confidence, classId)
        if (det.size() < 6) {
            std::cerr << "Invalid detection format. Expected at least 6 fields, got " << det.size() << "." << std::endl;
            continue;
        }

        // Extract bounding box and detection info
        int x = static_cast<int>(det[0]);
        int y = static_cast<int>(det[1]);
        int width = static_cast<int>(det[2]);
        int height = static_cast<int>(det[3]);
        float confidence = det[4];
        int classId = static_cast<int>(det[5]);

        // Validate classId against classLabels
        if (classId < 0 || classId >= static_cast<int>(classLabels.size())) {
            std::cerr << "Invalid classId: " << classId << ". Skipping detection." << std::endl;
            continue;
        }

        // Log the detection details
        std::cout << "Detection: "
                  << "Class: " << classLabels[classId]
                  << ", Confidence: " << confidence
                  << ", BBox: [" << x << ", " << y << ", " << width << ", " << height << "]"
                  << std::endl;

        // Draw the bounding box
        cv::rectangle(image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);

        // Prepare label text
        std::string label = classLabels[classId] + " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";

        // Draw the label background
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image, cv::Point(x, y - labelSize.height - 10), cv::Point(x + labelSize.width, y), cv::Scalar(0, 255, 0), cv::FILLED);

        // Put the label text
        cv::putText(image, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}
