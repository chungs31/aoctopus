#include "runtime.h"
#include "ocl_helper.h"
#include "AOCLUtils/aocl_utils.h"
#include <limits>

using namespace aocl_utils;

// The original run() function refactored from main. 
// Keep for multithreading later.
    //std::thread read_thread = std::thread(&Octokernel::copy_output_from_to_fcn, last, std::ref(d_y));
    /*for(unsigned i = 0; i < TEST_SET_SIZE; ++i) {
        if ((i+1) % 100 == 0 || i+1 == TEST_SET_SIZE) {
            printf("%5d/%d\r", i+1, TEST_SET_SIZE);
            fflush(stdout);
        }

        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(x_test[i]);

        // Enqueue all kernels in order.
        for (int k = 0; k < num_kernels; k++) {
            //if (k == num_kernels - 1 && read_thread.joinable()) { // last iter
            //    read_thread.join();
            //
            //}
            if (k == 3) {
                octokernels[k]->enqueue_kernel_reuse();
            }
            else if (k == 4) {
                octokernels[k]->enqueue_kernel(0);
            }
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
    */

/*
void Executor::run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) {
    Octokernel *last = octokernels[num_kernels- 1];
    for (unsigned i = 0; i < num_inputs; ++i) {
        //if ((i+1) % 100 == 0 || i+1 == num_inputs) {
        //    printf("%5d/%d\r", i+1, num_inputs);
        //    fflush(stdout);
        //}

        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(x_test[i]);

        // Enqueue all kernels in order.
        for (int k = 0; k < num_kernels; k++) {
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
    }
    printf("\n");
}
*/

/* compute argmax */
void Executor::predict(const scoped_array<scoped_aligned_ptr<float>> &d_y, scoped_array<int> &predictions) {
    for (int i = 0; i < num_inputs; i++) {
        float max_val = -std::numeric_limits<float>::infinity();
        int max_idx = -std::numeric_limits<int>::infinity();

        for (int output_idx = 0; output_idx < output_dim; output_idx++) {
            if (d_y[i][output_idx] > max_val) {
                max_val = d_y[i][output_idx];
                max_idx = output_idx;
            }
        }

        predictions[i] = max_idx;
    }
}

/* compare outputs and return number of non-equivalent preds */
int Executor::verify(const scoped_array<int> &y, const scoped_array<int> &y_ref) {
    int incorrect = 0;
    for (int i = 0; i < num_inputs; i++) {
        if (y[i] != y_ref[i]) incorrect++;
    }
    return incorrect;
}

/* is the accuracy good enough? */
bool Executor::pass(float accuracy) {
    if (accuracy > pass_threshold) return true;
    else return false;
}

/* MNIST Executor overriding functions */
int MNISTExecutor::map_weights() {
    int weight_idx = 0;
    for (int i = 0; i < num_kernels; i++) {
        Octokernel *kern = octokernels[i];
        int num_args = kern->get_n_bufs();
        if (num_args != -1) { // Skip layers without arguments.
            for (int j = 0; j < num_args; j++) {
                // If the buffer is labelled read-only and isn't an input layer (first layer),
                // then the layer contains parameters (weight/bias). Map to it.
                if (kern->buf_mflags[j] == CL_MEM_READ_ONLY && j != kern->get_input_idx()) {
                    printf("[DEBUG] LAYER %d: weight idx %d copied to HOSTMEM %d\n", i, weight_idx, j);
                    kern->load_buf(j, weights[weight_idx++]);
                }
            }
        }
    }
    return (weight_idx == weights.size());
}

void MNISTExecutor::run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) {
    Octokernel *last = octokernels[num_kernels- 1];
    for (unsigned i = 0; i < num_inputs; ++i) {
        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(x_test[i]);

        // Enqueue all kernels in order.
        for (int k = 0; k < num_kernels; k++) {
            if (type == MNISTExecutorType::REUSE) {
                if (k == 2) {
                    octokernels[k]->enqueue_kernel_reuse();
                }
                else {
                    octokernels[k]->enqueue_kernel();
                }
            }
            else {
                octokernels[k]->enqueue_kernel();
            }
        }

        // Copy output. Blocking call -- maybe multithread this later?
        last->copy_output_from_to(d_y[i]);
    }
    printf("\n");
}

/* MobileNet Executor overriding functions */
int MobileNetExecutor::map_weights() {
    int weight_idx = 0;
    for (int i = 0; i < num_kernels; i++) {
        Octokernel *kern = octokernels[i];
        int num_args = kern->get_n_bufs();

        switch (type) {
        case MobileNetExecutorType::CHANNELS:
            if (!kern->is_input_or_output_layer()) {
                for (int j = 0; j < num_args; j++) {
                    kern->load_buf(j, weights[weight_idx++]);
                }
            }
            break;
        default: // BASE
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
            break;
        }
    }
    return (weight_idx == weights.size());
}

void MobileNetExecutor::run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) {
    Octokernel *last = octokernels[num_kernels- 1];
    for (unsigned i = 0; i < num_inputs; ++i) {
        printf("%5d/%d\r", i+1, num_inputs);
        fflush(stdout);

        // Write input to host memory. Will be copied to buffer in enqueue.
        octokernels[0]->set_input_mem(x_test[i]);

        // Enqueue all kernels in order.
        for (int k = 0; k < num_kernels; k++) {
            octokernels[k]->enqueue_kernel();
            //octokernels[k]->dbg_dump_output();
        }

        // Copy output. Blocking call -- maybe multithread this later?
        last->copy_output_from_to(d_y[i]);
    }
    printf("\n");
}

