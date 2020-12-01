---
title: "Installation: Ubuntu"
---

# Ubuntu Installation

### For Ubuntu (>= 17.04)

**Installing pre-compiled Caffe**

Everything including caffe itself is packaged in 17.04 and higher versions.
To install pre-compiled Caffe package, just do it by

    sudo apt install caffe-cpu

for CPU-only version, or

    sudo apt install caffe-cuda

for CUDA version. Note, the cuda version may break if your NVIDIA driver
and CUDA toolkit are not installed by APT.

[Package status of CPU-only version](https://launchpad.net/ubuntu/+source/caffe)

[Package status of CUDA version](https://launchpad.net/ubuntu/+source/caffe-contrib)

**Installing Caffe from source**

We may install the dependencies by merely one line

    sudo apt build-dep caffe-cpu        # dependencies for CPU-only version
    sudo apt build-dep caffe-cuda       # dependencies for CUDA version

It requires a `deb-src` line in your `sources.list`.
Continue with [compilation](installation.html#compilation).

### For Ubuntu (\< 17.04)

**General dependencies**

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    sudo apt-get install --no-install-recommends libboost-all-dev
    sudo apt-get install libgflags-dev libgo