#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <chrono> // For timing

int main() {
    // Open the video file
    std::string videoPath = "../asset/test.mp4"; // Replace with your video file path
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file!" << std::endl;
        return -1;
    }

    // Get video properties
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << totalFrames << std::endl;

    // CUDA Canny edge detection parameters
    double lowThreshold = 50.0;
    double highThreshold = 150.0;

    // Timing variables
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = 0.0, gpuTime = 0.0;

    // =======================
    // CPU Processing
    // =======================
    std::cout << "Starting CPU processing..." << std::endl;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset video to the beginning
    cv::Mat frame, gray, edges;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalFrames; ++i) {
        if (!cap.read(frame)) break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply Canny edge detection on CPU
        cv::Canny(gray, edges, lowThreshold, highThreshold);
    }
    end = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<double>(end - start).count();
    std::cout << "CPU processing time: " << cpuTime << " seconds" << std::endl;

    // =======================
    // GPU Processing
    // =======================
    std::cout << "Starting GPU processing..." << std::endl;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset video to the beginning
    cv::cuda::GpuMat gpuFrame, gpuGray, gpuEdges;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalFrames; ++i) {
        if (!cap.read(frame)) break;

        // Upload frame to GPU
        gpuFrame.upload(frame);

        // Convert to grayscale
        cv::cuda::cvtColor(gpuFrame, gpuGray, cv::COLOR_BGR2GRAY);

        // Apply Canny edge detection on GPU
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(lowThreshold, highThreshold);
        canny->detect(gpuGray, gpuEdges);
    }
    end = std::chrono::high_resolution_clock::now();
    gpuTime = std::chrono::duration<double>(end - start).count();
    std::cout << "GPU processing time: " << gpuTime << " seconds" << std::endl;

    // =======================
    // Summary
    // =======================
    std::cout << "Comparison:" << std::endl;
    std::cout << "CPU Time: " << cpuTime << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpuTime << " seconds" << std::endl;
    std::cout << "Speedup (CPU/GPU): " << (cpuTime / gpuTime) << "x" << std::endl;

    // Release resources
    cap.release();
    return 0;
}
