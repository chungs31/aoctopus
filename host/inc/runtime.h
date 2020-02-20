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
#include "octokernel.h"

extern std::vector<Octokernel*> octokernels;
extern std::vector<std::vector<float> > weights; // imported weights from Keras
extern std::vector<std::vector<size_t> > bufsizes; // buffer sizes

extern aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > x_test;
extern aocl_utils::scoped_array<int> y_test;

enum class MNISTExecutorType {
    BASE,
    REUSE
};

class Executor {
public:
    int num_inputs;
    int output_dim;

    Executor(int n_i, int o_d) : num_inputs(n_i), output_dim(o_d) {};
    virtual ~Executor() {};

    // Following functions are specific to network architecture.
    virtual int map_weights() = 0;
    virtual void run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) = 0;

    void predict(const aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y, aocl_utils::scoped_array<int> &predictions);
    int verify(const aocl_utils::scoped_array<int> &y, const aocl_utils::scoped_array<int> &y_ref);
};

class MNISTExecutor : public Executor {
private:
    MNISTExecutorType type;
public:
    MNISTExecutor(int n_i, int o_d, MNISTExecutorType t) : Executor(n_i, o_d), type(t) {};
    virtual ~MNISTExecutor() {};
    virtual int map_weights() override;
    virtual void run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) override;
};


/*
class MobileNetExecutor : public Executor {
public:
    virtual void run() override;
};
*/

#endif /* RUNTIME_H */

