# YOLOv11 with Cuda and TensorRT

## Export model from pt to trt
```bash
pip install ultralytics
yolo export model=yolo11s.pt format=onnx batch=8 half=True

trtexec --onnx=yolo11s.onnx \
        --saveEngine=yolo11s.engine \
        --memPoolSize=workspace:4G \
        --fp16
```

## How to Build
### Build base image
```bash
docker build -t gtensorrt-opencv5-python3.11-cuda -f Dockerfile.base .
```

### Build inference image
```bash
docker build -t yolov11-cuda-trt -f Dockerfile .
```

## Run
### For workspace
```bash
docker run --gpus all -it --rm \
-v ./yolo-cuda:/workspace/yolo-cuda \
tensorrt-opencv5-python3.11-cuda bash
```

### For inference

```txt
Usage: ./build/main <input_path> [--engine_path=PATH] [--batch_size=N] [--confidence_threshold=FLOAT]
Example
./build/main ./asset/bus.jpg,./asset/bus1.jpg --engine_path=weights/yolo11s.engine
```

```bash
docker run --gpus all -it --rm \
-v ./weights:/workspace/weights \
-v ./asset:/workspace/asset \
yolov11-cuda-trt ./asset/bus.jpg --engine_path=weights/yolo11s.engine
```