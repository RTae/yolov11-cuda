## Dependency prep
```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev python3-dev \
    python3-numpy libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 ..

make -j$(nproc)
sudo make install
sudo ldconfig
```
## Build
```bash
rm -rf build && mkdir build && cd build
cmake .. && make
```

## Run
```bash
./main ../test.jpg
```