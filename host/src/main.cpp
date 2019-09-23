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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "importer.h"
#include "octokernel.h"

#include <vector>
#include <iostream>
#include <algorithm>

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements


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

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    // Import weights from Keras
    weight_parser("mnist_weight_dump.txt", weights);
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
    kernel.reset(num_devices);

    for(unsigned i = 0; i < num_devices; ++i) {
        // Command queue.
        queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue");

        // Kernel.
        const char *kernel_name = "fuse_dense_relu_kernel0";
        kernel[i] = clCreateKernel(program, kernel_name, &status);
        checkError(status, "Failed to create kernel");

        std::vector<size_t> buffer_sizes{400*sizeof(float), 48000*sizeof(float), 120*sizeof(float), 120*sizeof(float)};
        std::vector<cl_mem_flags> buffer_mflags{CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY};
        octokernels.push_back(new Octokernel(context, program, kernel_name, 4, buffer_sizes, buffer_mflags));
    }

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    if(num_devices == 0) {
        checkError(-1, "No devices");
    }

    ref_output.reset(weights[5].size());

    // Generate input
    for (int i = 0; i < 400; i++) {
        octokernels[0]->host_mems[0][i] = rand_float();
    }

    for (int i = 0; i < 48000; i++) {
        octokernels[0]->host_mems[1][i] = weights[4][i];
    }

    for (int i = 0; i < 120; i++) {
        octokernels[0]->host_mems[3][i] = weights[5][i];
    }
}

void run() {
    cl_int status;

    const double start_time = getCurrentTimestamp();

    // Launch the problem for each device.
    scoped_array<cl_event> kernel_event(num_devices);
    scoped_array<cl_event> finish_event(num_devices);

    for(unsigned i = 0; i < num_devices; ++i) {
        octokernels[0]->enqueue_kernel(queue[i]);
    }

    // Wait for all devices to finish.
    //const double end_time = getCurrentTimestamp();

    // Wall-clock time taken.
    //printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

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
    float sum;
    for (int ax1 = 0; ax1 < 120; ++ax1) {
        sum = 0.0;
        for (int k = 0; k < 400; ++k) {
            sum = (sum + (octokernels[0]->host_mems[0][k] * octokernels[0]->host_mems[1][((ax1 * 400) + k)]));
        }
        ref_output[ax1] = std::max((sum + octokernels[0]->host_mems[3][ax1]), 0.0e+00f);
    }
    bool pass = true;
    for(unsigned i = 0; i < num_devices && pass; ++i) {
        for(unsigned j = 0; j < 120 && pass; ++j) {
            if(fabsf(octokernels[0]->host_mems[2][j] - ref_output[j]) > 1.0e-5f) {
                printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
                        i, j, octokernels[0]->host_mems[2][j], ref_output[j]);
                pass = false;
            }
        }
    }

    printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
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

