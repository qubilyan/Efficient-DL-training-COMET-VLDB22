#include <vector>

#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  const int input_num_dims = bottom[0]->shape().size();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_dim = channel_axis_ + 1;
  num_spatial_axes_ = input_num_dims - first_spatial_dim;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << nu