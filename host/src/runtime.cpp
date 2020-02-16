#include "runtime.h"
#include "ocl_helper.h"
#include "AOCLUtils/aocl_utils.h"
#include <limits>

using namespace aocl_utils;

void Executor::run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) {
    Octokernel *last = octokernels[num_kernels- 1];
    for (unsigned i = 0; i < num_inputs; ++i) {
        if ((i+1) % 100 == 0 || i+1 == num_inputs) {
            printf("%5d/%d\r", i+1, num_inputs);
            fflush(stdout);
        }

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


