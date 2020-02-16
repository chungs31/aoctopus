#ifndef LAYER_H
#define LAYER_H

#include <vector>
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

#endif /* LAYER_H */

