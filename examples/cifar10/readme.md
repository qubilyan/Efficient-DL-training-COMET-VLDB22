---
title: CIFAR-10 tutorial
category: example
description: Train and test Caffe on CIFAR-10 data.
include_in_docs: true
priority: 5
---

Alex's CIFAR-10 tutorial, Caffe style
=====================================

Alex Krizhevsky's [cuda-convnet](https://code.google.com/p/cuda-convnet/) details the model definitions, parameters, and training procedure for good performance on CIFAR-10. This example reproduces his results in Caffe.

We will assume that you have Caffe successfully compiled. If not, please refer to the [Installation page](/installation.html). In this tutorial, we will assume that your caffe installation is located at `CAFFE_ROOT`.

We thank @chyojn for the pull request that defined the model schemas and solver configurations.

*This example is a work-in-progress. It would be nice to further explain details of the network and training choices and benchmark the full training.*

Prepare the Dataset
-------------------

You will first need to download and convert the data format from the [CIFAR-10 website](http://www.cs.toronto.edu/~kriz/cifar.html). To do this, simply run the following commands:

    cd $CAFFE_ROOT
    ./data/cifar10/get_cifar10.sh
    ./examples/cifar10/create_cifar10.sh

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be the dataset, `./cifar10-leveldb`, and the data set image mean `./mean.binaryproto`.

The Model
---------

The CIFAR-10 model is a CNN that composes layers of convolution, pooling, rectified linear unit (ReLU) nonlinearities, and local contrast normalization with a linear classifier on top of it all. We have defined the model in the `CAFFE_ROOT/examples/cifar10` directory's `cifar10_quick_train_test.prototxt`.

Training and Testing the "Quick" Model
--------------------------------------

Training the model is simple after you have written the network definition protobuf and solver protobuf files (refer to [MNIST Tutorial](../examples/mnist.html)). Simply run `train_quick.sh`, or the following command directly:

    cd $CAFFE_ROOT
    ./examples/cifar10/train_quick.sh

`train_quick.sh` is a simple script, so have a look inside. The main tool for training is `caffe` with the `train` action, and the solver protobuf text file as its argument.

When you run the code, you will see a lot of messages flying by like this:

    I0317 21:52:48.945710 2008298256 net.cpp:74] Creating Layer conv1
    I0317 21:52:48.945716 2008298256 net.cpp:84] conv1 <- data
    I0317 21:52:48.945725 2008298256 net.cpp:110] conv1 -> conv1
    I0317 21:52:49.298691 2008298256 net.cpp:125] Top shape: 100 32 32 32 (3276800)
    I0317 21:52:49.298719 2008298256 net.cpp:151] conv1 needs backward computation.

These messages tell you the details about each layer, its connections and its output shape, which may be helpful in debugging. After the initialization, the training will start:

    I0317 21:52:49.309370 2008298256 net.cpp:166] Network initialization done.
    I0317 21:52:49.309376 2008298256 net.cpp:167] Memory required for Data 23790808
    I0317 21:52:49.309422 2008298256 solver.cpp:36] Solver scaffolding done.
    I0317 21:52:49.309447 2008298256 solver.cpp:47] Solving CIFAR10_quick_train

Based on the solver setting, we will print the training loss function every 100 iterations, and test the network every 500 iterations. You will see messages like this:

    I0317 21:53:12.179772 2008298256 solver.cpp:208] Iteration 100, lr = 0.001
    I0317 21:53:12.185698 2008298256 solver.cpp:65] Iteration 100, loss = 1.73643
    ...
    I0317 21:54:41.150030 2008298256 solver.cpp:87] Iteration 500, Testing net
   