// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.


#include <vector>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <functional>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "importer.h"
#include "octokernel.h"
#include "lenet5.h"
#include "common.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_program program = NULL;

const cl_uint max_kernels_supported = 128;
cl_kernel kernels[max_kernels_supported];
int num_kernels; 

double wall_clock_time;

std::vector<Octokernel*> octokernels;
std::vector<std::vector<float> > weights; // imported weights from Keras

// 0 fuse_conv2d_relu_kernel0
// 1 fuse_avg_pool2d_kernel0 
// 2 fuse_conv2d_relu_1_kernel0
// 3 fuse_avg_pool2d_1_kernel0
// 4 fuse_transpose_flatten_kernel0
// 5 fuse_dense_relu_kernel0
// 6 fuse_dense_relu_1_kernel0
// 7 fuse_dense_kernel0
// 8 fuse_softmax_kernel0

// 0 54 
// 1 6
// 2 864
// 3 16
// 4 48000
// 5 120
// 6 10080
// 7 84
// 8 840
// 9 10

// Control whether the fast emulator should be used.
bool use_fast_emulator = false;
    
scoped_array<scoped_aligned_ptr<float> > mnist_x_test;
scoped_array<int> mnist_y_test;
scoped_array<int> d_y_test;

int TEST_SET_SIZE = 10000;

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();
void profiler_output();
//void pcie_bandwidth_test();

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);


    // Import weights from Keras
    weight_parser(config::file_weight, weights);
    printf("Weights imported: size %ld\n", weights.size());

    // Optional argument to specify the problem size.
    /*if(options.has("n")) {
      N = options.get<unsigned>("n");
      }*/

    // Optional argument to specify whether the fast emulator should be used.
    if(options.has("fast-emulator")) {
        use_fast_emulator = options.get<bool>("fast-emulator");
    }

    if(options.has("n")) {
        TEST_SET_SIZE = options.get<int>("n");
    }

    // Initialize OpenCL.
    if(!init_opencl()) {
        return -1;
    }

    // Initialize the problem data.
    // Requires the number of devices to be known.
    init_problem();
    
    // Run the kernel.
    run();

    // Free the resources allocated
    cleanup();

    // Print profiling times
    profiler_output();

    return 0;
}

/////// HELPER FUNCTIONS ///////
// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    if (use_fast_emulator) {
        platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    } else {
        platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    // Create the context.
    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    std::string binary_file = getBoardBinaryFile("aocl", device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Build the kernels now from the program
    status = clCreateKernelsInProgram(program, max_kernels_supported, kernels, (cl_uint *) &num_kernels);
    printf("Num kernels returned: %d\n", num_kernels);
    
    // If Intel Internal Autorun Profiling is on, have to decrease
    num_kernels--;

    for (int kern_id = 0; kern_id < num_kernels; kern_id++) {
        char func_name[128]; 
        cl_uint num_args;

        status = clGetKernelInfo(kernels[kern_id], CL_KERNEL_FUNCTION_NAME, 128, (void *) func_name, NULL);
        checkError(status, "Failed to get kernel func name");
        status = clGetKernelInfo(kernels[kern_id], CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL);
        checkError(status, "Failed to get kernel num args");

        printf("kernel name %s has %d arguments\n", func_name, num_args);
    }

    for(unsigned i = 0; i < num_devices; ++i) {
        // Kernel
        for (int kernel = 0; kernel < num_kernels; kernel++) {
            printf("Registering new kernel index %d named %s with %d bufs\n", kernel, config::cfg_network[kernel].func_name, config::cfg_network[kernel].n_bufs);

            octokernels.push_back(new Octokernel(
                context, 
                device[i],
                program,
                config::cfg_network[kernel].func_name, 
                //config::cfg_network[kernel].n_bufs, 
                config::cfg_network[kernel].buf_sizes, 
                config::cfg_network[kernel].buf_type, 
                config::cfg_network[kernel].output_layer_idx,
                config::cfg_network[kernel].input_layer_idx
            ));
            if (kernel > 0) {
                octokernels[kernel]->set_buffer_from_prev(octokernels[kernel - 1]);
            }
        }

        octokernels[0]->set_as_input_layer();
        octokernels[num_kernels - 1]->set_as_output_layer();
    }

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    if(num_devices == 0) {
        checkError(-1, "No devices");
    }

    import_mnist("../data/mnist_test.db", "../data/mnist_test_y.db", mnist_x_test, mnist_y_test);
   
    // Map weights to layers. Copy to read-only buffers that are not the input buffers.
    int weight_idx = 0;
    for (int i = 0; i < num_kernels; i++) {
        Octokernel *kern = octokernels[i];
        int num_args = kern->get_n_bufs();
        for (int j = 0; j < num_args; j++) {
            if (kern->buf_mflags[j] == CL_MEM_READ_ONLY && j != kern->get_input_idx()) {
                printf("LAYER %d: weight idx %d copied to HOSTMEM %d\n", i, weight_idx, j);
                kern->load_buf(j, weights[weight_idx++]);
            }
        }
    }
    assert(weight_idx == weights.size());
}

void run() {
    cl_int status;

    Octokernel *last = octokernels[num_kernels- 1];
    const double start_time = getCurrentTimestamp();
    
    // Copy the weights to global memory
    for (int k = 0; k < num_kernels; k++) {
        octokernels[k]->copy_weights_to_bufs();
        //if (k == 3) octokernels[k]->enqueue_kernel_reuse(1);
        //if (k == 4) octokernels[k]->enqueue_kernel(1);
        //octokernels[k]->enqueue_kernel(1); // only enable for globalmem
    }
    Octokernel::wait_for_write_queue();
    printf("Completed writing weights\n");

    scoped_array<scoped_aligned_ptr<float> > d_y;
    d_y.reset(TEST_SET_SIZE);
    for (int i = 0; i < TEST_SET_SIZE; i++) {
        d_y[i].reset(10);
    }

    //std::thread read_thread = std::thread(&Octokernel::copy_output_from_to_fcn, last, std::ref(d_y)); 
    for(unsigned i = 0; i < TEST_SET_SIZE; ++i) {
        if (i % 100 == 0) {
            printf("%5d/%d\r", i, TEST_SET_SIZE);
            fflush(stdout);
        }

        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(mnist_x_test[i]);

        // Enqueue all kernels in order.
        for (int k = 0; k < num_kernels; k++) {
            //if (k == num_kernels - 1 && read_thread.joinable()) { // last iter
            //    read_thread.join();
            //
            //}
            /* uncomment for reuse
            if (k == 3) { 
                octokernels[k]->enqueue_kernel_reuse();
            }
            else if (k == 4) {
                octokernels[k]->enqueue_kernel(0);
            }
            */
            if (k == 2) { 
                octokernels[k]->enqueue_kernel_reuse();
            }
            else {
                octokernels[k]->enqueue_kernel();//(0);
            }
            //octokernels[k]->dbg_dump_output();
        }

        // Copy output. Blocking call -- maybe multithread this later?
        last->copy_output_from_to(d_y[i]);
        //read_thread = std::thread(&Octokernel::copy_output_from_to, last, std::ref(d_y[i]));
        //read_thread.detach();
    }
    //printf("%d copied, %d ready\n", last->num_copied, last->num_ready);
    //read_thread.join();
    printf("\n");

    // Wait for all devices to finish.
    const double end_time = getCurrentTimestamp();

    // Wall-clock time taken.
    wall_clock_time = (end_time - start_time) * 1e3;

    scoped_array<int> predictions(TEST_SET_SIZE);

    // Verify
    int incorrect = 0;
    for (int i = 0; i < TEST_SET_SIZE; i++) {
        
        //printf("Prediction: \n");
        //printf("[");
        float max_val = -100000.0;
        int max_idx = -1000;
        for (int output_idx = 0; output_idx < 10; output_idx++) {
            /*
            if (output_idx != 9) {
                printf("%f, ", d_y[i][output_idx]);
            }
            else {
                printf("%f", d_y[i][output_idx]);
            }
            */
            
            if (d_y[i][output_idx] > max_val) {
                max_val = d_y[i][output_idx];
                max_idx = output_idx;
            }
        }
        predictions[i] = max_idx;
        
        //printf("]\n");
        //printf("Predicted number: %d\n", max_idx);
        
        if (predictions[i] != mnist_y_test[i]) {
            incorrect++;
        }
    }

    printf("Accuracy: %f\n", ((float)TEST_SET_SIZE - incorrect)/((float) TEST_SET_SIZE));
}



// Free the resources allocated during initialization
void cleanup() {
    for (auto obj : octokernels) {
        delete obj;
    }

    for (int i = 0; i < num_kernels; i++) {
        clReleaseKernel(kernels[i]);
    }

    if(program) {
        clReleaseProgram(program);
    }
    if(context) {
        clReleaseContext(context);
    }
}

void profiler_output() {
#ifdef OPENCL_PROFILER_ENABLE
    printf("OpenCL event profiler output\n");
    double sum = 0;
    for (int kernel = 0; kernel < num_kernels; kernel++) {
        sum += octokernels[kernel]->kernel_time;
        printf("Kernel %d execution time: %f ms\n", kernel, (double) octokernels[kernel]->kernel_time / 1000000.0);
    }
    printf("Total Kernel execution time: %f ms\n", (double) sum / 1000000.0);
    printf("Write to FPGA time: %f ms\n", (double) write_time / 1000000.0);
    printf("Read to FPGA time: %f ms\n", (double) read_time / 1000000.0);
    printf("Idle time: %f ms\n", wall_clock_time - (double)(sum+write_time+read_time)/1000000.0);
#endif
    printf("Wall clock time: %0.3f ms\n", wall_clock_time);
}

