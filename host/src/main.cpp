#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "common.h"
#include "importer.h"
#include "networks.h"
#include "octokernel.h"
#include "ocl_helper.h"
#include "runtime.h"

using namespace aocl_utils;

double wall_clock_time;

//scoped_array<int> d_y_test;
std::vector<Octokernel*> octokernels;
std::vector<std::vector<float> > weights; // imported weights from Keras
std::vector<std::vector<size_t> > bufsizes; // buffer sizes

scoped_array<scoped_aligned_ptr<float> > x_test;
scoped_array<int> y_test;

int TEST_SET_SIZE = 10000;

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    std::string user_cfg;
    if(options.has("c")) {
        user_cfg = options.get<std::string>("c");
        config::octocfg = config::select_config(user_cfg);
    }
    else {
        fprintf(stderr, "[ERROR] configuration unspecified\n");
        config::printcfgs();
        return -1;        
    }
    assert(config::octocfg /* Invalid configuration selected */);
    printf("[INFO] selected config %s\n", user_cfg.c_str());

    // Import weights from Keras
    weight_parser(config::octocfg->f_weight.c_str(), weights);
    printf("Weights imported: size %ld\n", weights.size());
    
    bufsizes_parser(config::octocfg->f_bufsizes.c_str(), bufsizes); // This is not weights but testing
    printf("Buf sizes imported: size %ld\n", bufsizes.size());

    // Optional argument to specify whether the fast emulator should be used.
    if(options.has("fast-emulator")) {
        use_fast_emulator = options.get<bool>("fast-emulator");
    }

    std::string bstream_path = "";
    if(options.has("e")) {
        putenv((char *)"CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1");
        printf("[INFO] CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA set to 1\n");
        bstream_path += "emul/";
    }

    if(options.has("n")) {
        TEST_SET_SIZE = options.get<int>("n");
    }

    // Initialize OpenCL.
    bstream_path += user_cfg;
    if(!init_opencl(bstream_path)) {
        return -1;
    }

    // Initialize the problem data.
    // Requires the number of devices to be known.
    init_problem();

    // Run the kernel.
    bool pass = run();

    // Free the resources allocated
    cleanup();

    // Free octokernels
    for (auto obj : octokernels) {
        delete obj;
    }

    // Print profiling times
    profiler_output();

    if (pass == true) return 0;
    return -1;
}

/////// HELPER FUNCTIONS ///////
// Initializes the OpenCL objects.
bool init_opencl(const std::string f_bitstream) {
    printf("Initializing OpenCL\n");

    cl_int status = init_opencl_internals(f_bitstream);
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
            printf("Registering new kernel index %d named %s with %d bufs\n", kernel, config::octocfg->cfg_network[kernel].func_name, config::octocfg->cfg_network[kernel].n_bufs);

            octokernels.push_back(new Octokernel(
                oclinfo.context,
                oclinfo.device[i],
                oclinfo.program,
                config::octocfg->cfg_network[kernel].func_name,
                // config::octocfg->cfg_network[kernel].n_bufs,
                config::octocfg->cfg_network[kernel].buf_sizes,
                //bufsizes[kernel],
                config::octocfg->cfg_network[kernel].buf_type,
                config::octocfg->cfg_network[kernel].output_layer_idx,
                config::octocfg->cfg_network[kernel].input_layer_idx
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

// Initialize the data for the problem. 
void init_problem() {
    if(oclinfo.num_devices == 0) {
        checkError(-1, "No devices");
    }

    // Get input data
    config::octocfg->importer.import_input_data(x_test, y_test);    /* for real data */
    //config::octocfg->importer->generate_random_input(10000, x_test); /* for random data */

    // Map weights to layers.
    int check = config::octocfg->executor->map_weights();
    assert(check);
}

bool run() {
    cl_int status;
    config::octocfg->executor->num_inputs = TEST_SET_SIZE;                     // Set number of inputs

    Octokernel *last = octokernels[num_kernels- 1];
    const double start_time = getCurrentTimestamp();

    // Copy the weights to global memory
    for (int k = 0; k < num_kernels; k++) {
        // MOBILENET
        //if (!octokernels[k]->is_input_or_output_layer())
        //    octokernels[k]->copy_weights_to_bufs();
        octokernels[k]->copy_weights_to_bufs();
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
    config::octocfg->executor->run(d_y);

    // Wait for all devices to finish.
    const double end_time = getCurrentTimestamp();

    // Wall-clock time taken.
    wall_clock_time = (end_time - exec_time) * 1e3;

    scoped_array<int> predictions(TEST_SET_SIZE);

    // Verify
    config::octocfg->executor->predict(d_y, predictions);                      // Calculate predictions
    int incorrect = config::octocfg->executor->verify(predictions, y_test);    // Compare predictions to reference
    float accuracy = ((float)TEST_SET_SIZE - incorrect)/((float) TEST_SET_SIZE);    
    printf("[INFO] Accuracy: %f\n", accuracy);

    // Is it above threshold
    bool pass = config::octocfg->executor->pass(accuracy);
    printf("*** VALIDATION %s\n", (pass ? ("PASSED") : ("FAILED")));
    return pass;
}

