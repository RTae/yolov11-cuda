#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

// Detection struct
struct Detection {
    int class_id;  // Class ID
    float conf;    // Confidence score
    cv::Rect bbox; // Bounding box

    // Constructor for ease of initialization
    Detection(int classId, float confidence, const cv::Rect& box)
        : class_id(classId), conf(confidence), bbox(box) {}
};

std::vector<Detection> postprocess(
    const float* output, int numDetections, int numClasses,
    float confThreshold, float nmsThreshold);

// Draw detections on an image
void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, const std::vector<std::string>& classLabels);

#endif // POSTPROCESSOR_H