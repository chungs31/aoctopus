#ifndef IMAGENET_H
#define IMAGENET_H

#include "layer.h" 

namespace ImageNet {

Layer SqueezeNet[] {
    {"fuse_conv2d_relu_kernel0", -1, {}, {}, -1, -1},
    {"fuse_max_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_max_pool2d_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_5_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_6_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_7_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_8_kernel0", -1, {}, {}, -1, -1},
    {"fuse_max_pool2d_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_9_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_10_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_11_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_12_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_13_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_14_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_15_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_16_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_17_kernel0", -1, {}, {}, -1, -1},
    {"fuse_global_avg_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_transpose_flatten_kernel0", -1, {}, {}, -1, -1},
    {"fuse_softmax_kernel0", -1, {}, {}, -1, -1}
};

}

#endif 

