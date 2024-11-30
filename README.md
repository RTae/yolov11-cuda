# YOLOv11 with CUDA and TensorRT

This project provides a high-performance implementation of YOLOv11 object detection using TensorRT for inference acceleration. The pipeline processes images and videos in batches, leveraging CUDA for preprocessing and inference. Non-Maximum Suppression (NMS) and postprocessing are performed on the CPU to optimize results.

![Walk Video Example](https://github.com/RTae/yolov11-cuda/blob/main/asset/mat_walk.gif?raw=true)

![Bus Example](https://github.com/RTae/yolov11-cuda/blob/main/asset/mat_bus.jpg?raw=true)
## Features

- **CUDA-accelerated preprocessing and inference** for faster performance.
- **Batch processing** for images and videos, supporting multiple inputs simultaneously.
- **Postprocessing with NMS** to refine detections.
- **Threaded Execution** to handle multiple input streams concurrently, with each stream processed on separate CUDA streams for maximum GPU utilization.
- **Scalable pipeline**: Process multiple files (images or videos) in parallel.
- Outputs detections with bounding boxes, class labels, and confidence scores.

## Model Preparation

### Export the YOLOv11 Model to TensorRT Engine

1. Install the necessary tools for exporting the model:
   ```bash
   pip install ultralytics
   ```

2. Convert the PyTorch YOLO model to ONNX format:
   ```bash
   yolo export model=yolo11s.pt format=onnx batch=8 half=True
   ```

3. Compile the ONNX model into a TensorRT engine:
   ```bash
   trtexec --onnx=yolo11s.onnx \
           --saveEngine=yolo11s.engine \
           --memPoolSize=workspace:4G \
           --fp16
   ```

   - `--onnx`: Specifies the ONNX model file.
   - `--saveEngine`: Specifies the output TensorRT engine file.
   - `--memPoolSize`: Allocates GPU memory for the engine.
   - `--fp16`: Enables half-precision floating-point computation for faster inference.

## Build Instructions

### 1. Build Base Docker Image
The base Docker image includes TensorRT, OpenCV, and Python 3.11:
```bash
docker build -t tensorrt-opencv5-python3.11-cuda -f Dockerfile.base .
```

### 2. Build the Inference Image
The inference Docker image includes the YOLOv11 pipeline:
```bash
docker build -t yolov11-cuda-trt -f Dockerfile .
```

## Running the Project

### 1. Development Environment
To enter an interactive environment for development:
```bash
docker run --gpus all -it --rm \
-v $(pwd)/yolo-cuda:/workspace/yolo-cuda \
tensorrt-opencv5-python3.11-cuda bash
```

### 2. Inference

#### Usage
Run the inference executable with the following options:
```txt
Usage: ./build/main <input_path> [--engine_path=PATH] [--batch_size=N] [--confidence_threshold=FLOAT]
Example:
./build/main ./asset/walk1.mp4,./asset/walk2.mp4 --engine_path=weights/yolo11s.engine --batch_size=8 --confidence_threshold=0.7
```

- `<input_path>`: Comma-separated list of input image or video paths.
- `--engine_path` (optional): Path to the TensorRT engine file (default: `./weights/yolo11s.engine`).
- `--batch_size` (optional): Number of inputs to process per batch (default: `8`).
- `--confidence_threshold` (optional): Confidence threshold for filtering detections (default: `0.7`).

#### Running Inference with Docker
To run inference using the Docker image:
```bash
docker run --gpus all -it --rm \
-v $(pwd)/weights:/workspace/weights \
-v $(pwd)/asset:/workspace/asset \
yolov11-cuda-trt ./asset/walk1.mp4,./asset/walk2.mp4 --engine_path=weights/yolo11s.engine --batch_size=8 --confidence_threshold=0.7
```

## Pipeline Details

### 1. Preprocessing (CUDA)
- **Resizes and normalizes** input images or video frames to 640x640 resolution.
- Converts color space from BGR to RGB.
- **Batch processing**: Combines multiple inputs for parallel GPU processing.
- **Format Conversion**: Converts images to NCHW format.

### 2. Inference (TensorRT)
- Executes the TensorRT engine on the GPU.
- Processes inputs in batches for efficiency.
- **Leverages CUDA streams** to overlap computation and data transfer.

### 3. Postprocessing (CPU)
- **Confidence Filtering**: Removes low-confidence detections based on a threshold.
- **Non-Maximum Suppression (NMS)**: Removes overlapping bounding boxes for the same object.
- Outputs detections with:
  - **Class IDs**
  - **Confidence scores**
  - **Bounding box coordinates**

### 4. Multi-threaded Execution
- **Threaded Inference**: Input files (images or videos) are processed concurrently using multiple threads.
- **CUDA Streams**: Each thread operates on a separate CUDA stream to parallelize preprocessing, inference, and data transfer for multiple inputs.

### 5. Logging and Output
- **Logs inference times** for each batch, frame, and individual input file.
- Logs detections with class labels, confidence scores, and bounding box details.
- Saves processed images and videos with bounding boxes drawn.

## Example Outputs

1. **Image Input**:
   - Input: `./asset/bus.jpg`
   - Command:
     ```bash
     ./build/main ./asset/bus.jpg --engine_path=weights/yolo11s.engine --confidence_threshold=0.8
     ```
   - Output: Annotated image saved as `out_bus.jpg`.

2. **Video Input**:
   - Input: `./asset/walk1.mp4`
   - Command:
     ```bash
     ./build/main ./asset/walk1.mp4 --engine_path=weights/yolo11s.engine --batch_size=4 --confidence_threshold=0.7
     ```
   - Output: Annotated video saved as `out_walk1.mp4`.

## Performance

The pipeline processes inputs efficiently by leveraging GPU acceleration. Below are approximate performance metrics:
- **Preprocessing and inference**: Executed on the GPU for faster computation.
- **Postprocessing**: Executed on the CPU for flexibility and precision.
- **Throughput**: Supports batch sizes up to the GPU memory limit, providing high throughput for both images and videos.
- **Multi-threading**: Achieves concurrent processing of multiple inputs, significantly improving throughput.

## Limitations

- **Postprocessing** is CPU-bound, which may bottleneck performance for large batch sizes.
- Requires a TensorRT-compatible GPU.