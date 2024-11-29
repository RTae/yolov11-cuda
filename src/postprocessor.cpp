#include "postprocessor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>

// Helper function for Non-Maximum Suppression (NMS)
std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iouThreshold) {
    std::vector<int> indices;
    std::vector<int> sortedIndices(scores.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    // Sort scores in descending order
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&scores](int i, int j) {
        return scores[i] > scores[j];
    });

    while (!sortedIndices.empty()) {
        int current = sortedIndices.front();
        indices.push_back(current);
        sortedIndices.erase(sortedIndices.begin());

        auto it = std::remove_if(sortedIndices.begin(), sortedIndices.end(), [&](int idx) {
            float iou = (float)(boxes[current] & boxes[idx]).area() / (float)(boxes[current] | boxes[idx]).area();
            return iou > iouThreshold;
        });
        sortedIndices.erase(it, sortedIndices.end());
    }

    return indices;
}

// Postprocess YOLO output
std::vector<std::vector<float>> postprocess(const float* output, int batchSize, int numClasses, int numAnchors, float confThreshold, float iouThreshold) {
    std::vector<std::vector<float>> detections;

    for (int b = 0; b < batchSize; ++b) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int anchor = 0; anchor < numAnchors; ++anchor) {
            const float* anchorData = output + b * (numClasses + 5) * numAnchors + anchor * (numClasses + 5);

            // Bounding box attributes
            float x = anchorData[0];  // Center x
            float y = anchorData[1];  // Center y
            float w = anchorData[2];  // Width
            float h = anchorData[3];  // Height
            float objectness = anchorData[4];  // Objectness score

            // Class probabilities
            const float* classScores = anchorData + 5;
            int maxClassId = std::distance(classScores, std::max_element(classScores, classScores + numClasses));
            float classConfidence = classScores[maxClassId];

            // Final confidence = objectness * class confidence
            float confidence = objectness * classConfidence;

            if (confidence < confThreshold) continue;

            // Convert center coordinates to top-left corner
            int x1 = static_cast<int>(x - w / 2.0f);
            int y1 = static_cast<int>(y - h / 2.0f);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Size(width, height)));
            confidences.push_back(confidence);
            classIds.push_back(maxClassId);
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> nmsIndices = nms(boxes, confidences, iouThreshold);

        for (int idx : nmsIndices) {
            const auto& box = boxes[idx];
            detections.push_back({
                (float)box.x, (float)box.y, (float)box.width, (float)box.height, confidences[idx], (float)classIds[idx]
            });
        }
    }

    return detections;
}

void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, const std::vector<std::string>& classLabels) {
    for (const auto& det : detections) {
        int x = static_cast<int>(det[0]);
        int y = static_cast<int>(det[1]);
        int width = static_cast<int>(det[2]);
        int height = static_cast<int>(det[3]);
        float confidence = det[4];
        int classId = static_cast<int>(det[5]);

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
