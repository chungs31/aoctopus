/* 2020-02-13 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * runtime.h
 *
 * Class defining the executor and verifier functions.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include "AOCLUtils/aocl_utils.h"

class Executor {
public:
    int num_inputs;
    int output_dim;

    Executor(int n_i, int o_d) : num_inputs(n_i), output_dim(o_d) {};

    //virtual void run() = 0;
    void predict(const aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y, aocl_utils::scoped_array<int> &predictions);
    int verify(const aocl_utils::scoped_array<int> &y, const aocl_utils::scoped_array<int> &y_ref);
};

/*
class MNISTExecutor : public Executor {
public:
    virtual void run() override;
};

class MobileNetExecutor : public Executor {
public:
    virtual void run() override;
};
*/

#endif /* RUNTIME_H */

