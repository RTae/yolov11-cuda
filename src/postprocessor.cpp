#include "postprocessor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

std::vector<Detection> postprocess(
    const float* output, int numCandidates, int numClasses, float confThreshold, float nmsThreshold) {
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    // Process each candidate box
    for (int i = 0; i < numCandidates; ++i) {
        // Bounding box attributes
        float x = output[i * (numClasses + 4)];
        float y = output[i * (numClasses + 4) + 1];
        float w = output[i * (numClasses + 4) + 2];
        float h = output[i * (numClasses + 4) + 3];

        // Class scores start at index (4)
        const float* classScores = output + i * (numClasses + 4) + 4;

        // Find the class with the maximum score
        int maxClassId = std::distance(classScores, std::max_element(classScores, classScores + numClasses));
        float confidence = classScores[maxClassId];

        // Check if the confidence is above the threshold
        if (confidence >= confThreshold) {
            // Convert (cx, cy, w, h) to top-left corner (x1, y1) and width/height
            int x1 = static_cast<int>(x - w / 2.0f);
            int y1 = static_cast<int>(y - h / 2.0f);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Size(width, height)));
            confidences.push_back(confidence);
            classIds.push_back(maxClassId);
        }
    }

    // Apply Non-Maximum Suppression (NMS) using OpenCV
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    for (int idx : nmsIndices) {
        detections.push_back({
            classIds[idx],         // Class ID
            confidences[idx],      // Confidence score
            boxes[idx]             // Bounding box
        });
    }

    // Log detections
    std::cout << "Post-NMS Detections: " << detections.size() << std::endl;
    for (const auto& det : detections) {
        std::cout << "Detection: "
                  << "Class ID: " << det.class_id
                  << ", Confidence: " << det.conf
                  << ", BBox: [" << det.bbox.x << ", " << det.bbox.y
                  << ", " << det.bbox.width << ", " << det.bbox.height << "]"
                  << std::endl;
    }

    return detections;
}

void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classLabels) {
    for (const auto& det : detections) {
        // Extract bounding box information
        int x = det.bbox.x;
        int y = det.bbox.y;
        int width = det.bbox.width;
        int height = det.bbox.height;
        float confidence = det.conf;
        int classId = det.class_id;

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
