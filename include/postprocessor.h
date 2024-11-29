#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    int class_id;
    float conf;
    cv::Rect bbox;
};

std::vector<Detection> postprocess(const float* output, int numAnchors, int numClasses, float confThreshold, float nmsThreshold);

// Function to draw detections on an image
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classLabels);

#endif // POSTPROCESSOR_H