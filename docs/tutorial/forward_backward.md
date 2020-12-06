---
title: Forward and Backward for Inference and Learning
---
# Forward and Backward

The forward and backward passes are the essential computations of a [Net](net_layer_blob.html).

<img src="fig/forward_backward.png" alt="Forward and Backward" width="480">

Let's consider a simple logistic regression classifier.

The **forward** pass computes the output given the input for inference.
In forward Caffe composes the computation of each layer to compute the "function" represented by the model.
This pass goes from bottom to top.

<img src="fig/forward.jpg" alt="Forward pass" width="320">

The data $$x$$ is passed through an inner product layer for $$g(x)$$ then through a softmax for $$h(g(x))$$ and softmax loss to give $$f_W(x)$$.

The **backward** pass computes the gradient given the loss for learning.
In backward Caffe reverse-composes the gradient of each layer to compute the gradient of the whole model by automatic differentiation.
This is back-propagation.
This pass goes from top to bottom.

<img src="fig/backward.jpg" alt="Backward pass" width="320">

The backward pass begins with the loss and computes the gradient with respect to the output $