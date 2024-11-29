#include "postprocessor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>


std::vector<Detection> postprocess(
    const float* output, 
    int numDetections, 
    int numClasses, 
    float confThreshold, 
    float nmsThreshold) {
    
    std::vector<cv::Rect> boxes;          // Bounding boxes
    std::vector<int> classIds;            // Class IDs
    std::vector<float> confidences;       // Confidence scores
    std::vector<Detection> results;       // Final detections after NMS

    // The format of the model output is: [numClasses+4, numDetections]
    cv::Mat detOutput(numClasses + 4, numDetections, CV_32F, const_cast<float*>(output));

    for (int i = 0; i < detOutput.cols; ++i) {
        // Extract class scores for the current detection
        cv::Mat classScores = detOutput.col(i).rowRange(4, 4 + numClasses);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(classScores, nullptr, &maxClassScore, nullptr, &classIdPoint);

        // Check if the confidence score exceeds the threshold
        if (maxClassScore > confThreshold) {
            // Extract bounding box coordinates
            float cx = detOutput.at<float>(0, i);  // Center X
            float cy = detOutput.at<float>(1, i);  // Center Y
            float ow = detOutput.at<float>(2, i);  // Width
            float oh = detOutput.at<float>(3, i);  // Height

            // Convert to top-left corner and dimensions
            int x = static_cast<int>(cx - 0.5 * ow);
            int y = static_cast<int>(cy - 0.5 * oh);
            int width = static_cast<int>(ow);
            int height = static_cast<int>(oh);

            // Store bounding box, class ID, and confidence
            boxes.emplace_back(cv::Rect(cv::Point(x, y), cv::Size(width, height)));
            classIds.push_back(classIdPoint.y);
            confidences.push_back(static_cast<float>(maxClassScore));
        }
    }

    // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    // Populate results with detections after NMS
    for (int idx : nmsIndices) {
        Detection detection;
        detection.class_id = classIds[idx];
        detection.conf = confidences[idx];
        detection.bbox = boxes[idx];

        // Log final detection details
        std::cout << "[Final Detection] "
                  << "Class ID: " << detection.class_id 
                  << ", Confidence: " << detection.conf
                  << ", BBox: [" << detection.bbox.x << ", " 
                  << detection.bbox.y << ", " 
                  << detection.bbox.width << ", " 
                  << detection.bbox.height << "]" << std::endl;

        results.push_back(detection);
    }

    return results;
}

void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classLabels, int input_w, int input_h) {
    // Calculate the scaling ratios between input dimensions and original image dimensions
    const float ratio_h = input_h / static_cast<float>(image.rows);
    const float ratio_w = input_w / static_cast<float>(image.cols);

    for (const auto& det : detections) {
        // Extract bounding box information
        auto box = det.bbox;
        int classId = det.class_id;
        float confidence = det.conf;

        // Adjust bounding box coordinates based on aspect ratio
        if (ratio_h > ratio_w) {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        } else {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        // Ensure bounding box dimensions stay within image bounds
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(box.width, image.cols - box.x);
        box.height = std::min(box.height, image.rows - box.y);

        // Draw the bounding box on the image
        cv::rectangle(image, cv::Rect(box.x, box.y, box.width, box.height), cv::Scalar(0, 255, 0), 2);

        // Prepare the label text
        std::string label = classLabels[classId] + " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";

        // Calculate the size of the label background
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Define the label background rectangle
        cv::rectangle(image, 
                      cv::Point(box.x, box.y - labelSize.height - 10), 
                      cv::Point(box.x + labelSize.width, box.y), 
                      cv::Scalar(0, 255, 0), 
                      cv::FILLED);

        // Put the label text on the image
        cv::putText(image, label, 
                    cv::Point(box.x, box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0, 0, 0), 1);
    }
}