# Stage 1: Build Stage
FROM tensorrt-opencv5-python3.11-cuda AS builder

# Set working directory
WORKDIR /workspace

# Copy source files and CMakeLists.txt to the container
COPY ./src /workspace/src
COPY ./include /workspace/include
COPY CMakeLists.txt /workspace
COPY main.cu /workspace

# Create a build directory and compile the project
RUN mkdir -p build && cd build && cmake .. && make

# Stage 2: Runtime Stage
FROM tensorrt-opencv5-python3.11-cuda

# Set working directory
WORKDIR /workspace

# Copy compiled binaries from the build stage
COPY --from=builder /workspace/build/main /workspace/main

# Default entrypoint to run the executable
ENTRYPOINT ["./main"]
