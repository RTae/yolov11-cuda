#include "postprocessor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

// Postprocess the inference output to extract detections
std::vector<Detection> postprocess(
    const float* output, 
    int numAnchors, 
    int numClasses, 
    float confThreshold, 
    float nmsThreshold) {

    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    std::cout << "Starting postprocess..." << std::endl;

    // Iterate over all anchors
    for (int anchor = 0; anchor < numAnchors; ++anchor) {
        // Extract bounding box attributes
        float xCenter = output[anchor];                              // Center x
        float yCenter = output[numAnchors + anchor];                 // Center y
        float width = output[2 * numAnchors + anchor];               // Width
        float height = output[3 * numAnchors + anchor];              // Height
        float objectness = output[4 * numAnchors + anchor];          // Objectness score

        // Extract class probabilities
        const float* classScores = output + 5 * numAnchors + anchor * numClasses;

        // Find the class with the maximum probability
        cv::Point maxClassIdPoint;
        double maxClassScore;
        cv::minMaxLoc(cv::Mat(1, numClasses, CV_32F, (void*)classScores), nullptr, &maxClassScore, nullptr, &maxClassIdPoint);

        // Compute the final confidence
        float confidence = objectness * static_cast<float>(maxClassScore);

        // Log confidence scores and bounding box details
        std::cout << "Anchor: " << anchor 
                  << ", Objectness: " << objectness 
                  << ", Max Class Confidence: " << maxClassScore 
                  << ", Final Confidence: " << confidence 
                  << ", BBox: [" << xCenter << ", " << yCenter << ", " 
                  << width << ", " << height << "]" 
                  << std::endl;

        if (confidence < confThreshold) {
            continue; // Skip low-confidence detections
        }

        // Convert (center x, center y, width, height) to (x, y, width, height)
        int x = static_cast<int>(xCenter - width / 2.0f);
        int y = static_cast<int>(yCenter - height / 2.0f);
        int w = static_cast<int>(width);
        int h = static_cast<int>(height);

        // Store results
        boxes.emplace_back(x, y, w, h);
        confidences.push_back(confidence);
        classIds.push_back(maxClassIdPoint.x);
    }

    std::cout << "Total pre-NMS detections: " << confidences.size() << std::endl;

    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Extract NMS results
    for (int idx : indices) {
        detections.push_back(Detection{classIds[idx], confidences[idx], boxes[idx]});
    }

    std::cout << "Total post-NMS detections: " << detections.size() << std::endl;

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
