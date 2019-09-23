/* 2019-09-17 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * lenet5.h
 *
 * LeNet buffer definitions related to ../device/aocl.cl.
 */

#include <vector>
#include <stdlib.h>
#include "CL/opencl.h"

// Alias of CL mem types for convenience
enum BUF_TYPE {
    rw = CL_MEM_READ_WRITE,
    r = CL_MEM_READ_ONLY,
    w = CL_MEM_WRITE_ONLY
};

struct Layer {
    const char *func_name;
    int n_bufs;
    std::vector<size_t> buf_sizes; // Num of elements, i.e., NOT in bytes
    std::vector<cl_mem_flags> buf_type;  
    int output_layer_idx;
    int input_layer_idx;
};

namespace LeNet5 {

const int num_layers = 9;

Layer network[] {
    {"fuse_conv2d_relu_kernel0", 5, {676, 784, 54, 4056, 6}, {w, r, r, w, r}, 3, 1}, 
    {"fuse_avg_pool2d_kernel0", 2, {1014, 4056}, {w, r}, 0, 1}, 
    {"fuse_conv2d_relu_1_kernel0", 5, {121, 1014, 864, 1936, 16}, {w, r, r, w, r}, 3, 1},
    {"fuse_avg_pool2d_1_kernel0", 2, {400, 1924}, {w, r}, 0, 1}, 
    {"fuse_transpose_flatten_kernel0", 2, {400, 400}, {w, r}, 0, 1},
    {"fuse_dense_relu_kernel0", 4, {400, 48000, 120, 120}, {r, r, w, r}, 2, 0},
    {"fuse_dense_relu_1_kernel0", 5, {1, 120, 10080, 84, 84}, {w, r, r, w, r}, 3, 1},
    {"fuse_dense_kernel0", 5, {1, 84, 840, 10, 10}, {w, r, r, w, r}, 3, 1},
    {"fuse_softmax_kernel0", 4, {1, 10, 1, 10}, {w, r, w, w}, 3, 1}
};

}

