#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

// Perform Non-Maximum Suppression (NMS) to filter overlapping detections
std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iouThreshold);

// Postprocess YOLO output to extract bounding boxes, class IDs, and confidence scores
std::vector<std::vector<float>> postprocess(const float* output, int batchSize, int numClasses, int numAnchors, float confThreshold, float iouThreshold);

// Draw detections on an image
void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, const std::vector<std::string>& classLabels);

#endif