#include "runtime.h"
#include "AOCLUtils/aocl_utils.h"
#include <limits>

using namespace aocl_utils;

static void predict(const scoped_array<scoped_aligned_ptr<float>> &d_y, scoped_array<int> &predictions);
    
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


