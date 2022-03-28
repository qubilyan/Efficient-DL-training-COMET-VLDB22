#include <algorithm>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/bnll_layer.hpp"
#include "caffe/layers/clip_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/elu_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/log_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/swish_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NeuronLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NeuronLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestDropoutForward(const float dropout_ratio) {
    LayerParameter layer_param;
    // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
    // set it explicitly to test that 0.5 is the default.
    if (dropout_ratio != 0.5) {
      layer_param.mutable_dropout_param()->set_dropout_ratio(dropout_ratio);
    }
    DropoutLayer<Dtype> layer(layer_param);
    layer_param.set_phase(TRAIN);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    float scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
    const int count = this->blob_bottom_->count();
    // Initialize num_kept to count the number of inputs NOT dropped out.
    int num_kept = 0;
    for (int i = 0; i < count; ++i) {
      if (top_data[i] != 0) {
        ++num_kept;
        EXPECT_EQ(top_data[i], bottom_data[i] * scale);
      }
    }
    const Dtype std_error = sqrt(dropout_ratio * (1 - dropout_ratio) / count);
    // Fail if the number dropped was more than 1.96 * std_error away from the
    // expected number -- requires 95% confidence that the dropout layer is not
    // obeying the given dropout_ratio for test failure.
    const Dtype empirical_dropout_ratio = 1 - num_kept / Dtype(count);
    EXPECT_NEAR(empirical_dropout_ratio, dropout_ratio, 1.96 * std_error);
  }

  void TestExpForward(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const Dtype kDelta = 2e-4;
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const Dtype bottom_val = bottom_data[i];
      const Dtype top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(top_val, exp(shift + scale * bottom_val), kDelta);
      } else {
        EXPECT_NEAR(top_val, pow(base, shift + scale * bottom_val), kDelta);
      }
    }
  }

  void TestExpGradient(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }

  void TestPReLU(PReLULayer<Dtype> *layer) {
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype* slope_data = layer->blobs()[0]->cpu_data();
    int hw = this->blob_bottom_->height() * this->blob_bottom_->width();
    int channels = this->blob_bottom_->channels();
    bool channel_shared = layer->layer_param().prelu_param().channel_shared();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      int c = channel_shared ? 0 : (i / hw) % channels;
      EXPECT_EQ(top_data[i],
          std::max(bottom_data[i], (Dtype)(0))
          + slope_data[c] * std::min(bottom_data[i], (Dtype)(0)));
    }
  }

  void LogBottomInit() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
    caffe_exp(this->blob_bottom_->count(), bottom_data, bottom_data);
  }

  void TestLogForward(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const Dtype kDelta = 2e-4;
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const Dtype bottom_val = bottom_data[i];
      const Dtype top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(top_val, log(shift + scale * bottom_val), kDelta);
      } else {
        EXPECT_NEAR(top_val, log(shift + scale * bottom_val) / log(base),
                    kDelta);
      }
    }
  }

  void TestLogGradient(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }
};

TYPED_TEST_CASE(NeuronLayerTest, TestDtypesAndDevices);

TYPED_TEST(NeuronLayerTest, TestAbsVal) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data    = this->blob_top_->cpu_data();
  const int count = this->blob_bottom_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(top_data[i], fabs(bottom_data[i]));
  }
}

TYPED_TEST(NeuronLayerTest, TestAbsGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestClip) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "clip_param { min: -1, max: 2 }", &layer_param));
  ClipLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], -1);
    EXPECT_LE(top_data[i], 2);
    EXPECT_TRUE(bottom_data[i] > -1 || top_data[i] == -1);
    EXPECT_TRUE(bottom_data[i] < 2 || top_data[i] == 2);
    EXPECT_TRUE(!(bottom_data[i] >= -1 && bottom_data[i] <= 2)
            || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(NeuronLayerTest, TestClipGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "clip_param { min: -1, max: 2 }", &layer_param));
  ClipLayer<Dtype> layer(layer_param);
  // Unfortunately, it might happen that an input value lands exactly within
  // the discontinuity region of the Clip function. In this case the numeric
  // gradient is likely to differ significantly (i.e. by a value larger than
  // checker tolerance) from the computed gradient. To handle such cases, we
  // eliminate such values from the input blob before the gradient check.
  const Dtype epsilon = 1e-2;
  const Dtype min_range_start = layer_param.clip_param().min() - epsilon;
  const Dtype min_range_end   = layer_param.clip_param().min() + epsilon;
  const Dtype max_range_start = layer_param.clip_param().max() - epsilon;
  const Dtype max_range_end   = layer_param.clip_param().max() + epsilon;
  // The input blob is owned by the NeuronLayerTest object, so we begin with
  // creating a temporary blob and copying the input data there.
  Blob<Dtype> temp_bottom;
  temp_bottom.ReshapeLike(*this->blob_bottom_);
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  Dtype* temp_data_mutable = temp_bottom.mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (bottom_data[i] >= min_range_start &&
        bottom_data[i] <= min_range_end) {
      temp_data_mutable[i] = bottom_data[i] - epsilon;
    } else if (bottom_data[i] >= max_range_start &&
               bottom_data[i] <= max_range_end) {
      temp_data_mutable[i] = bottom_data[i] + epsilon;
    } else {
      temp_data_mutable[i] = bottom_data[i];
    }
  }
  vector<Blob<Dtype>*> temp_bottom_vec;
  temp_bottom_vec.push_back(&temp_bottom);
  GradientChecker<Dtype> checker(epsilon, 1e-3);
  checker.CheckGradientEltwise(&layer, temp_bottom_vec, this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestReLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(Neu