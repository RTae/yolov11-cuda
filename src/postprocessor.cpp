#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "postprocessor.h"

std::vector<Detection> postprocess(
    const float* output, int numDetections, int numClasses,
    float confThreshold, float nmsThreshold) {
    std::vector<cv::Rect> boxes;          // Bounding boxes
    std::vector<int> class_ids;           // Class IDs
    std::vector<float> confidences;       // Confidence scores
    std::vector<Detection> detections;    // Final detection results

    for (int i = 0; i < numDetections; ++i) {
        const float* detectionData = output + i * (numClasses + 4); // Access detection data
        float cx = detectionData[0];  // Center x
        float cy = detectionData[1];  // Center y
        float ow = detectionData[2];  // Width
        float oh = detectionData[3];  // Height

        const float* classScores = detectionData + 4; // Class scores
        int maxClassId = std::max_element(classScores, classScores + numClasses) - classScores;
        float classConfidence = classScores[maxClassId];

        float confidence = classConfidence; // Use class confidence directly

        if (confidence > confThreshold) {
            int x = static_cast<int>(cx - ow / 2.0f);
            int y = static_cast<int>(cy - oh / 2.0f);
            int width = static_cast<int>(ow);
            int height = static_cast<int>(oh);

            boxes.emplace_back(cv::Rect(x, y, width, height));
            class_ids.push_back(maxClassId);
            confidences.push_back(confidence);
        }
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    for (int idx : nmsIndices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.conf = confidences[idx];
        det.bbox = boxes[idx];
        detections.push_back(det);
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
