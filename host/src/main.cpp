#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "importer.h"
#include "octokernel.h"
#include "lenet5.h"
#include "common.h"
#include "ocl_helper.h"

using namespace aocl_utils;

double wall_clock_time;

std::vector<Octokernel*> octokernels;
std::vector<std::vector<float> > weights; // imported weights from Keras
std::vector<std::vector<size_t> > bufsizes; // buffer sizes

scoped_array<scoped_aligned_ptr<float> > x_test;
scoped_array<int> y_test;
//scoped_array<int> d_y_test;

int TEST_SET_SIZE = 10000;

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    // Import weights from Keras
    weight_parser(config::file_weight, weights);
    printf("Weights imported: size %ld\n", weights.size());
    
    bufsizes_parser(config::file_bufsizes, bufsizes); // This is not weights but testing
    printf("Buf sizes imported: size %ld\n", bufsizes.size());

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

    // Free octokernels
    for (auto obj : octokernels) {
        delete obj;
    }

    // Print profiling times
    profiler_output();

    return 0;
}

/////// HELPER FUNCTIONS ///////
// Initializes the OpenCL objects.
bool init_opencl() {
    printf("Initializing OpenCL\n");

    cl_int status = init_opencl_internals();
    checkError(status, "Failed to initialize OpenCL internals");

    // Build the kernels now from the oclinfo.program
    status = clCreateKernelsInProgram(oclinfo.program, max_kernels_supported, kernels, (cl_uint *) &num_kernels);
    printf("Num kernels returned: %d\n", num_kernels);

    // If Intel Internal Autorun Profiling is on, have to decrease
#ifdef INTEL_PROFILER_ENABLE
    num_kernels--;
#endif

    for (int kern_id = 0; kern_id < num_kernels; kern_id++) {
        char func_name[128];
        cl_uint num_args;

        status = clGetKernelInfo(kernels[kern_id], CL_KERNEL_FUNCTION_NAME, 128, (void *) func_name, NULL);
        checkError(status, "Failed to get kernel func name");
        status = clGetKernelInfo(kernels[kern_id], CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL);
        checkError(status, "Failed to get kernel num args");

        printf("kernel name %s has %d arguments\n", func_name, num_args);
    }

    for(unsigned i = 0; i < oclinfo.num_devices; ++i) {
        // Kernel
        for (int kernel = 0; kernel < num_kernels; kernel++) {
            printf("Registering new kernel index %d named %s with %d bufs\n", kernel, config::cfg_network[kernel].func_name, config::cfg_network[kernel].n_bufs);

            octokernels.push_back(new Octokernel(
                oclinfo.context,
                oclinfo.device[i],
                oclinfo.program,
                config::cfg_network[kernel].func_name,
                // config::cfg_network[kernel].n_bufs,
                config::cfg_network[kernel].buf_sizes,
                //bufsizes[kernel],
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

// Initialize the data for the problem. Requires oclinfo.num_devices to be known.
void init_problem() {
    if(oclinfo.num_devices == 0) {
        checkError(-1, "No devices");
    }


    config::octocfg->importer->import_input_data(x_test, y_test);    /* for real data */
    //config::octocfg->importer->generate_random_input(10000, x_test); /* for random data */

    // Map weights to layers. Copy to read-only buffers that are not the input buffers.
    // Works for LeNet5...
    int weight_idx = 0;
    for (int i = 0; i < num_kernels; i++) {
        Octokernel *kern = octokernels[i];
        int num_args = kern->get_n_bufs();
        if (num_args != -1) { // Manual-ish configuration.
            for (int j = 0; j < num_args; j++) {
                if (kern->buf_mflags[j] == CL_MEM_READ_ONLY && j != kern->get_input_idx()) {
                    printf("LAYER %d: weight idx %d copied to HOSTMEM %d\n", i, weight_idx, j);
                    kern->load_buf(j, weights[weight_idx++]);
                }
            }
        }

        // MOBILENET

        //else { */

        // THIS IS THE DEFAULT MOBILENET CONFIGUROR
        /*
            if (num_args < 5) {
                // Only kernels with 5 arguments have weights/biases.
                // Don't load them.
                continue;
            }
            else {
                kern->load_buf(2, weights[weight_idx++]);
                kern->load_buf(4, weights[weight_idx++]);
                if (num_args > 5) {
                  kern->load_buf(5, weights[weight_idx++]);
                }
            }
        */
        // MOBILENET CHANNELS CONFIGUROR

        /*
        if (!kern->is_input_or_output_layer()) {
            for (int j = 0; j < num_args; j++) {
                kern->load_buf(j, weights[weight_idx++]);
            }
        }
        */


        //}
    }

    // All weights should have been mapped.
    assert(weight_idx == weights.size());
}

void run() {
    cl_int status;

    Octokernel *last = octokernels[num_kernels- 1];
    const double start_time = getCurrentTimestamp();

    // Copy the weights to global memory
    for (int k = 0; k < num_kernels; k++) {
        // MOBILENET
        //if (!octokernels[k]->is_input_or_output_layer())
        //    octokernels[k]->copy_weights_to_bufs();
        octokernels[k]->copy_weights_to_bufs();

        // Reuse stuff
        //if (k == 3) octokernels[k]->enqueue_kernel_reuse(1);
        //if (k == 4) octokernels[k]->enqueue_kernel(1);
        //octokernels[k]->enqueue_kernel(1); // only enable for globalmem
    }
    Octokernel::wait_for_write_queue();
    printf("Completed writing weights\n");

    scoped_array<scoped_aligned_ptr<float> > d_y;
    d_y.reset(TEST_SET_SIZE);
    size_t output_size = last->get_buf_size(last->get_output_idx());
    for (int i = 0; i < TEST_SET_SIZE; i++) {
        // Allocate space for last output to be copied here.
        d_y[i].reset(output_size);
    }

    const double exec_time = getCurrentTimestamp();
    //std::thread read_thread = std::thread(&Octokernel::copy_output_from_to_fcn, last, std::ref(d_y));
    for(unsigned i = 0; i < TEST_SET_SIZE; ++i) {
        //if (i % 100 == 0) {
            printf("%5d/%d\r", i, TEST_SET_SIZE);
            fflush(stdout);
        //}

        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(x_test[i]);

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
    wall_clock_time = (end_time - exec_time) * 1e3;

    scoped_array<int> predictions(TEST_SET_SIZE);

    // Verify
    int incorrect = 0;
    for (int i = 0; i < TEST_SET_SIZE; i++) {

        //printf("Prediction: \n");
        //printf("[");
        float max_val = -100000.0;
        int max_idx = -1000;
        for (int output_idx = 0; output_idx < output_size; output_idx++) {
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
        //printf("Predicted class: %d\n", max_idx);

        // Verification step: skip for now
        if (predictions[i] != y_test[i]) {
            incorrect++;
        }
    }

    printf("Accuracy: %f\n", ((float)TEST_SET_SIZE - incorrect)/((float) TEST_SET_SIZE));
}
