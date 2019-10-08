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

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "importer.h"
#include "octokernel.h"
#include "lenet5.h"
#include "common.h"

#define TEST_SET_SIZE 10000

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements

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

// Problem data
scoped_array<float> ref_output; // num_devices elements

// Control whether the fast emulator should be used.
bool use_fast_emulator = false;
    
scoped_array<scoped_aligned_ptr<float> > mnist_x_test;
scoped_array<int> mnist_y_test;
scoped_array<int> d_y_test;

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();
void profiler_output();

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    // Import weights from Keras
    weight_parser("../data/mnist_weight_dump.txt", weights);
    printf("Weights imported: size %ld\n", weights.size());

    // Optional argument to specify the problem size.
    /*if(options.has("n")) {
      N = options.get<unsigned>("n");
      }*/

    // Optional argument to specify whether the fast emulator should be used.
    if(options.has("fast-emulator")) {
        use_fast_emulator = options.get<bool>("fast-emulator");
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

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

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

    // Create per-device objects.
    queue.reset(num_devices);
    //kernel.reset(num_devices);

    for(unsigned i = 0; i < num_devices; ++i) {
        // Command queue.
        queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue");

        // Kernel
        for (int kernel = 0; kernel < LeNet5::num_layers; kernel++) {
            printf("Registering new kernel index %d named %s with %d bufs\n", kernel, LeNet5::network[kernel].func_name, LeNet5::network[kernel].n_bufs);
            octokernels.push_back(new Octokernel(
                context, 
                device[i],
                program,
                LeNet5::network[kernel].func_name, 
                LeNet5::network[kernel].n_bufs, 
                LeNet5::network[kernel].buf_sizes, 
                LeNet5::network[kernel].buf_type, 
                LeNet5::network[kernel].output_layer_idx,
                LeNet5::network[kernel].input_layer_idx
            ));
            if (kernel > 0) {
                octokernels[kernel]->set_buffer_from_prev(octokernels[kernel - 1]);
            }
        }

        octokernels[0]->set_as_input_layer();
        octokernels[LeNet5::num_layers - 1]->set_as_output_layer();
    }

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    if(num_devices == 0) {
        checkError(-1, "No devices");
    }

    import_mnist("../data/mnist_test.db", "../data/mnist_test_y.db", mnist_x_test, mnist_y_test);
    //ref_output.reset(4056);

    // Load weights to host memory
    printf("Copying weights to host memory for layer 0\n");
    octokernels[0]->load_buf(2, weights[0]);
    octokernels[0]->load_buf(4, weights[1]);
    
    printf("Copying weights to host memory for layer 2\n");
    octokernels[2]->load_buf(2, weights[2]);
    octokernels[2]->load_buf(4, weights[3]);
    
    printf("Copying weights to host memory for layer 5\n");
    octokernels[5]->load_buf(2, weights[4]);
    octokernels[5]->load_buf(4, weights[5]);

    printf("Copying weights to host memory for layer 6\n");
    octokernels[6]->load_buf(2, weights[6]);
    octokernels[6]->load_buf(4, weights[7]);
    
    printf("Copying weights to host memory for layer 7\n");
    octokernels[7]->load_buf(2, weights[8]);
    octokernels[7]->load_buf(4, weights[9]);
    
    /* Channels
    printf("Copying weights to host memory for layer 0\n");
    octokernels[0]->load_buf(1, weights[0]);
    octokernels[0]->load_buf(2, weights[1]);
    
    printf("Copying weights to host memory for layer 2\n");
    octokernels[2]->load_buf(0, weights[2]);
    octokernels[2]->load_buf(1, weights[3]);
    
    printf("Copying weights to host memory for layer 5\n");
    octokernels[5]->load_buf(0, weights[4]);
    octokernels[5]->load_buf(1, weights[5]);

    printf("Copying weights to host memory for layer 6\n");
    octokernels[6]->load_buf(0, weights[6]);
    octokernels[6]->load_buf(1, weights[7]);
    
    printf("Copying weights to host memory for layer 7\n");
    octokernels[7]->load_buf(0, weights[8]);
    octokernels[7]->load_buf(1, weights[9]);
    */
}

void run() {
    cl_int status;

    Octokernel *last = octokernels[LeNet5::num_layers- 1];
    const double start_time = getCurrentTimestamp();
    
    // Copy the weights to global memory
    for (int k = 0; k < LeNet5::num_layers; k++) {
        octokernels[k]->copy_weights_to_bufs();
    }

    scoped_array<scoped_array<float> > d_y;
    d_y.reset(TEST_SET_SIZE);

    //cl_event prev = NULL, nullev = NULL, dummy = NULL;

    for(unsigned i = 0; i < TEST_SET_SIZE; ++i) {
        if (i % 100 == 0) {
            printf("Processing iteration %d\n", i);
        }

        octokernels[0]->set_input_mem(mnist_x_test[i]);
        //prev = octokernels[0]->enqueue_kernel(prev);
        octokernels[0]->enqueue_kernel();
        //octokernels[0]->dbg_dump_outpt();

        for (int k = 1; k < LeNet5::num_layers; k++) {
            //octokernels[k]->set_input_mem(octokernels[k-1]->host_mems[octokernels[k-1]->get_output_idx()]);
            octokernels[k]->enqueue_kernel();
            //clReleaseEvent(dummy);
            //octokernels[k]->dbg_dump_output();
        }

        last->copy_output_from_to(d_y[i]);
    }

    // Wait for all devices to finish.
    const double end_time = getCurrentTimestamp();

    // Wall-clock time taken.
    wall_clock_time = (end_time - start_time) * 1e3;
    printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

    scoped_array<int> predictions(TEST_SET_SIZE);

    // Verify
    int incorrect = 0;
    for (int i = 0; i < TEST_SET_SIZE; i++) {
        printf("Prediction: \n");
        printf("[");
        float max_val = -100000.0;
        int max_idx = -1000;
        for (int output_idx = 0; output_idx < 10; output_idx++) {
            if (output_idx != 9) {
                printf("%f, ", d_y[i][output_idx]);
            }
            else {
                printf("%f", d_y[i][output_idx]);
            }
            
            if (d_y[i][output_idx] > max_val) {
                max_val = d_y[i][output_idx];
                max_idx = output_idx;
            }
        }
        predictions[i] = max_idx;
        
        printf("]\n");
        printf("Predicted number: %d\n", max_idx);
        
        if (predictions[i] != mnist_y_test[i]) {
            incorrect++;
        }
    }


    printf("Accuracy: %f\n", ((float)TEST_SET_SIZE - incorrect)/((float) TEST_SET_SIZE));

    // Get kernel times using the OpenCL event profiling API.
    /*for(unsigned i = 0; i < num_devices; ++i) {
        cl_ulong time_ns = getStartEndTime(kernel_event[i]);
        printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
    }*/

    // Release all events.
    /*
    for(unsigned i = 0; i < num_devices; ++i) {
        clReleaseEvent(kernel_event[i]);
        clReleaseEvent(finish_event[i]);
    }
    */

    // Verify results.
    /*float sum;
    for (int ax1 = 0; ax1 < 120; ++ax1) {
        sum = 0.0;
        for (int k = 0; k < 400; ++k) {
            sum = (sum + (octokernels[5]->host_mems[0][k] * octokernels[5]->host_mems[1][((ax1 * 400) + k)]));
        }
        ref_output[ax1] = std::max((sum + octokernels[5]->host_mems[3][ax1]), 0.0e+00f);
    }
    bool pass = true;
    for(unsigned i = 0; i < num_devices && pass; ++i) {
        for(unsigned j = 0; j < 120 && pass; ++j) {
            if(fabsf(octokernels[5]->host_mems[2][j] - ref_output[j]) > 1.0e-5f) {
                printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
                        i, j, octokernels[5]->host_mems[2][j], ref_output[j]);
                pass = false;
            }
        }
    }*/

    // Verifying first laye
    /*
    float compute[676];

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 26; ++yy) {
      for (int xx = 0; xx < 26; ++xx) {
        compute[((yy * 26) + xx)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            compute[((yy * 26) + xx)] = (compute[((yy * 26) + xx)] + (mnist_x_test[0][((((yy + ry) * 28) + xx) + rx)] * octokernels[0]->host_mems[2][((((ax1 * 3) + ry) * 3) + rx)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      for (int ax3 = 0; ax3 < 26; ++ax3) {
        ref_output[((((ax1 * 26) + ax2) * 26) + ax3)] = std::max((compute[((ax2 * 26) + ax3)] + octokernels[0]->host_mems[4][ax1]), 0.000000e+00f);
      }
    }
  }
    bool pass = true;
    for(unsigned i = 0; i < num_devices && pass; ++i) {
        for(unsigned j = 0; j < 4056 && pass; ++j) {
            if(fabsf(octokernels[0]->host_mems[3][j] - ref_output[j]) > 1.0e-5f) {
                printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
                        i, j, octokernels[0]->host_mems[3][j], ref_output[j]);
                pass = false;
            }
        }
    }
    */
}



// Free the resources allocated during initialization
void cleanup() {
    for (auto obj : octokernels) {
        delete obj;
    }

    for(unsigned i = 0; i < num_devices; ++i) {
        if(queue && queue[i]) {
            clReleaseCommandQueue(queue[i]);
        }
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
    printf("Kernel execution time: %f ms\n", (double) kernel_time / 1000000.0);
    printf("Write to FPGA time: %f ms\n", (double) write_time / 1000000.0);
    printf("Read to FPGA time: %f ms\n", (double) read_time / 1000000.0);
    printf("Idle time: %f ms\n", wall_clock_time - (double)(kernel_time+write_time+read_time)/1000000.0);
#endif
    printf("Wall clock time: %f ms\n", wall_clock_time);
}

