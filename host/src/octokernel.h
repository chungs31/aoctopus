/* 2019-09-18 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * octokernel.h
 *
 * Define the octokernel class. 
 *
 * Create cl_kernel objects and buffers related to the kernel (layer).
 *
 */
#pragma once

#include <string>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

class Octokernel {
public:
    // Kernels
    cl_kernel kernel;
    std::string kernel_name;

    // CL Buffers
    scoped_array<cl_mem> bufs;
    scoped_array<size_t> buf_lens;
    int n_bufs;

    // Host memory
    scoped_array<scoped_aligned_ptr<float> > host_mems;
    
    // Public functions
    Octokernel(cl_context context,
               cl_program program, 
               const char *_kernel_name, 
               int num_buffers, 
               scoped_array<size_t> buffer_sizes,
               scoped_array<cl_mem_flags> buffer_mflags);
    ~Octokernel();

    void copy_host_to_bufs();

};

Octokernel::Octokernel(cl_context context, cl_program program, const char *_kernel_name, int num_buffers, scoped_array<size_t> buffer_sizes, scoped_array<cl_mem_flags> buffer_mflags) {
    // Store name of kernel
    kernel_name = _kernel_name;  

    // Initialize kernel
    cl_int status;
    kernel = clCreateKernel(program, _kernel_name, &status);
    checkError(status, "Failed to create kernel");
    
    n_bufs = num_buffers;
    host_mems.reset(num_buffers);
    
    for (int i = 0; i < num_buffers; i++) {
        // Initialize CL buffers
        buf_lens[i] = buffer_sizes[i];
        bufs[i] = clCreateBuffer(context, buffer_mflags[i], sizeof(float) * buffer_sizes[i], NULL, &status);
        checkError(status, "Failed to create buffer for bias");

        // Initialize CPU variables
        host_mems[i].reset(buffer_sizes[i]);
    }
}

Octokernel::~Octokernel() {
    clReleaseKernel(kernel);
    for (int i = 0; i < n_bufs; i++) {
        clReleaseMemObject(bufs[i]);
    }
}

