# COMET: A New Memory-Efficient Deep Learning Training Framework Using Error-Bounded Lossy Compression

COMET is a modified version of the [Caffe](https://github.com/BVLC/caffe) framework, enabling memory-efficient deep learning training through error-bounded lossy compression technology, [SZ](https://github.com/szcompressor/SZ). We've primarily focused on modifying the Caffe [1] layer function to support SZ [2] for compressing activation data.

**Note:** This repository now is maintained at new account: qubilyan.

## Setup Methods

There are two methods for setting up and using COMET.

### Method 1: Use Docker Image (Recommended)

To simplify the use of COMET, we provide a docker image with the essential environment.

#### Step 1: Pull the docker image

Assuming [docker](https://docs.docker.com/get-docker/) has been installed, run this command to pull our docker image from DockerHub:
```
docker pull jinsian/caffe
```

#### Step 2: Training with COMET

First, launch the docker image:
```
docker run -ti jinsian/caffe:COMET /bin/bash
```

Then, use the following command to start the training process with AlexNet on the Stanford Dogs dataset with COMET:
```
cd /opt/caffe
./build/tools/caffe train -solver ./models/bvlc_reference_caffenet/solver.prototxt
```

### Method 2: Build From Source

#### Step 1: Clone the repo and install SZ

Use this command to clone the repo:
```
git clone https://github.com/qubilyan/Efficient-DL-training-COMET-VLDB22
```

Install SZ following instructions shown at https://github.com/szcompressor/SZ before building COMET.

#### Step 2: Compile Caffe for COMET

#### Step 3: Download and prepare dataset

#### Step 4: Training with COMET

## References

[1] Yangqing Jia, et al. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the 22nd ACM international conference on Multimedia, pp. 675-678. 2014.

[2] Tian, Jiannan, et al. "cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data." In Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques, pp. 3-15. 2020.