#include "networks.h"

namespace LeNet5 {

Layer base_network[] { // Default network (Baseline, Unrolled)
    {"fuse_conv2d_relu_kernel0", 5, {676, 784, 54, 4056, 6}, {rw, r, r, w, r}, 3, 1}, 
    {"fuse_avg_pool2d_kernel0", 2, {1014, 4056}, {w, r}, 0, 1}, 
    {"fuse_conv2d_relu_1_kernel0", 5, {121, 1014, 864, 1936, 16}, {w, r, r, w, r}, 3, 1},
    {"fuse_avg_pool2d_1_kernel0", 2, {400, 1924}, {w, r}, 0, 1}, 
    {"fuse_transpose_flatten_kernel0", 2, {400, 400}, {w, r}, 0, 1},
    {"fuse_dense_relu_kernel0", 5, {1, 400, 48000, 120, 120}, {w, r, r, w, r}, 3, 1},
    {"fuse_dense_relu_1_kernel0", 5, {1, 120, 10080, 84, 84}, {w, r, r, w, r}, 3, 1},
    {"fuse_dense_kernel0", 5, {1, 84, 840, 10, 10}, {w, r, r, w, r}, 3, 1},
    {"fuse_softmax_kernel0", 4, {1, 10, 1, 10}, {w, r, w, w}, 3, 1}
};

Layer channels_network[] { // Channels
    {"fuse_conv2d_relu_kernel0", 3, {784, 54, 6}, {r, r, r}, -1, 0}, 
    {"fuse_avg_pool2d_kernel0", 0, {}, {}, -1, -1}, 
    {"fuse_conv2d_relu_1_kernel0", 2, {864, 16}, {r, r}, -1, -1},
    {"fuse_avg_pool2d_1_kernel0", 0, {}, {}, -1, -1}, 
    {"fuse_transpose_flatten_kernel0", 0, {}, {}, -1, -1},
    {"fuse_dense_relu_kernel0", 2, {48000, 120}, {r, r}, -1, -1},
    {"fuse_dense_relu_1_kernel0", 2, {10080, 84}, {r, r}, -1, -1},
    {"fuse_dense_kernel0", 2, {840, 10}, {r, r}, -1, -1},
    {"fuse_softmax_kernel0", 1, {10}, {w}, 0, -1}
};

Layer autorun_network[] { // Channels + Autorun Network
    {"fuse_conv2d_relu_kernel0", 3, {784, 54, 6}, {r, r, r}, -1, 0}, 
    {"fuse_conv2d_relu_1_kernel0", 2, {864, 16}, {r, r}, -1, -1},
    {"fuse_dense_relu_kernel0", 2, {48000, 120}, {r, r}, -1, -1},
    {"fuse_dense_relu_1_kernel0", 2, {10080, 84}, {r, r}, -1, -1},
    {"fuse_dense_kernel0", 2, {840, 10}, {r, r}, -1, -1},
    {"fuse_softmax_kernel0", 1, {10}, {w}, 0, -1}
};

Layer reuse_network[] { // Channels + Autorun Network
    {"fuse_conv2d_relu_kernel0", 3, {784, 54, 6}, {r, r, r}, -1, 0}, 
    {"fuse_conv2d_relu_1_kernel0", 2, {864, 16}, {r, r}, -1, -1},
    {"read_to_ch4", 1, {400}, {rw}, 0, -1},
    {"fuse_dense_reuse", 10, {400, 48000, 120, 10080, 84, 840, 10, 120, 84, 10}, 
        {r, r, r, r, r, r, r, rw, rw, rw}, 9, 0},
    {"write_to_ch7", 1, {10}, {rw}, -1, 0},
    {"fuse_softmax_kernel0", 1, {10}, {w}, 0, -1}
};

Layer reuse_fused_network[] { // Channels + Autorun Network
    {"fuse_conv2d_relu_kernel0", 3, {784, 54, 6}, {r, r, r}, -1, 0}, 
    {"fuse_conv2d_relu_1_kernel0", 2, {864, 16}, {r, r}, -1, -1},
    {"fuse_dense_reuse", 10, {400, 48000, 120, 10080, 84, 840, 10, 120, 84, 10}, 
        {rw, r, r, r, r, r, r, rw, rw, rw}, 9, 0},
    {"fuse_softmax_kernel0", 1, {10}, {w}, 0, -1}
};

Layer combined_dense_network[] { // Channels + Autorun Network
    {"fuse_conv2d_relu_kernel0", 3, {784, 54, 6}, {r, r, r}, -1, 0}, 
    {"fuse_conv2d_relu_1_kernel0", 2, {864, 16}, {r, r}, -1, -1},
    {"read_to_ch4", 1, {400}, {rw}, 0, -1},
    {"fuse_dense_combined", 10, {400, 48000, 120, 10080, 84, 840, 10, 120, 84, 10}, 
        {r, r, r, r, r, r, r, rw, rw, rw}, 9, 0},
    {"write_to_ch7", 1, {10}, {rw}, -1, 0},
    {"fuse_softmax_kernel0", 1, {10}, {w}, 0, -1}
};

Layer globalmem_network[] { 
    {"fuse_conv2d_relu_kernel0", 4, {784, 54, 6, 4056}, {r, r, r, w}, 3, 0}, 
    {"fuse_avg_pool2d_kernel0", 2, {4056, 1014}, {r, w}, 1, 0}, 
    {"fuse_conv2d_relu_1_kernel0", 4, {1014, 864, 16, 1936}, {r, r, r, w}, 3, 0},
    {"fuse_avg_pool2d_1_kernel0", 2, {1924, 400}, {r, w}, 1, 0}, 
    {"fuse_transpose_flatten_kernel0", 2, {400, 400}, {r, w}, 1, 0},
    {"fuse_dense_relu_kernel0", 4, {400, 48000, 120, 120}, {r, r, r, w}, 3, 0},
    {"fuse_dense_relu_1_kernel0", 4, {120, 10080, 84, 84}, {r, r, r, w}, 3, 0},
    {"fuse_dense_kernel0", 4, {84, 840, 10, 10}, {r, r, r, w}, 3, 0},
    {"fuse_softmax_kernel0", 2, {10, 10}, {r, w}, 1, 0}
};

}

namespace ImageNet {

Layer SqueezeNet[] {
    {"fuse_conv2d_relu_kernel0", -1, {}, {}, -1, 1},
    {"fuse_max_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_2_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_3_kernel1", -1, {}, {}, -1, -1},
    {"fuse_concatenate_kernel1", -1, {}, {}, -1, -1},
    {"fuse_max_pool2d_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_5_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_6_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_7_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_8_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_6_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_7_kernel1", -1, {}, {}, -1, -1},
    {"fuse_concatenate_1_kernel1", -1, {}, {}, -1, -1},
    {"fuse_max_pool2d_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_9_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_10_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_11_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_12_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_10_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_11_kernel1", -1, {}, {}, -1, -1},
    {"fuse_concatenate_2_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_13_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_14_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_15_kernel0", -1, {}, {}, -1, -1},
    {"fuse_concatenate_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_16_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_14_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_15_kernel1", -1, {}, {}, -1, -1},
    {"fuse_concatenate_3_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_relu_17_kernel0", -1, {}, {}, -1, -1},
    {"fuse_global_avg_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_transpose_flatten_kernel0", -1, {}, {}, -1, -1},
    {"fuse_softmax_kernel0", -1, {}, {}, 3, -1}
};

Layer MobileNet[] {
    {"fuse_pad_kernel0", -1, {}, {}, -1, 1},
    {"fuse_conv2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_5_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_6_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_7_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_8_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_9_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_10_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_11_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_12_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel2", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel2", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel3", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel3", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel4", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel4", -1, {}, {}, -1, -1},
    {"fuse_pad_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_15_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_16_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_17_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_18_kernel0", -1, {}, {}, -1, -1},
    {"fuse_global_avg_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_transpose_flatten_reshape_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_kernel1", -1, {}, {}, -1, -1},
    {"fuse_reshape_kernel0", -1, {}, {}, -1, -1},
    {"fuse_softmax_kernel0", -1, {}, {}, 3, -1}
};

Layer MobileNet_channels[] {
    {"fuse_pad_kernel0", -1, {}, {}, -1, 0},
    {"fuse_conv2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_1_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_5_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_6_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_2_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_7_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_8_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_9_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_10_kernel0", -1, {}, {}, -1, -1},
    {"fuse_pad_3_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_11_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_12_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel1", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel2", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel2", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel3", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel3", -1, {}, {}, -1, -1},
    {"fuse_conv2d_13_kernel4", -1, {}, {}, -1, -1},
    {"fuse_conv2d_14_kernel4", -1, {}, {}, -1, -1},
    {"fuse_pad_4_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_15_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_16_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_17_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_18_kernel0", -1, {}, {}, -1, -1},
    {"fuse_global_avg_pool2d_kernel0", -1, {}, {}, -1, -1},
    {"fuse_transpose_flatten_reshape_kernel0", -1, {}, {}, -1, -1},
    {"fuse_conv2d_kernel1", -1, {}, {}, -1, -1},
    {"fuse_reshape_kernel0", -1, {}, {}, -1, -1},
    {"fuse_softmax_kernel0", -1, {}, {}, 0, -1}
};

}
