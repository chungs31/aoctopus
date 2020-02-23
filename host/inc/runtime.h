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

enum class MobileNetExecutorType {
    BASE,
    CHANNELS
};

class Executor {
public:
    int num_inputs;
    int output_dim;
    float pass_threshold;

    Executor(int n_i, int o_d, float p_t) : num_inputs(n_i), output_dim(o_d), pass_threshold(p_t) {};
    virtual ~Executor() {};

    // Following functions are specific to network architecture.
    virtual int map_weights() = 0;
    virtual void run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) = 0;

    // Common functions for all executors.
    void predict(const aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y, aocl_utils::scoped_array<int> &predictions);
    int verify(const aocl_utils::scoped_array<int> &y, const aocl_utils::scoped_array<int> &y_ref);
    bool pass(float accuracy);
};

class MNISTExecutor : public Executor {
private:
    MNISTExecutorType type;
public:
    MNISTExecutor(int n_i, int o_d, float p_t, MNISTExecutorType t) : Executor(n_i, o_d, p_t), type(t) {};
    virtual ~MNISTExecutor() {};
    virtual int map_weights() override;
    virtual void run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) override;
};


class MobileNetExecutor : public Executor {
private: 
    MobileNetExecutorType type;
public:
    MobileNetExecutor(int n_i, int o_d, float p_t, MobileNetExecutorType t) : Executor(n_i, o_d, p_t), type(t) {};
    virtual ~MobileNetExecutor() {};
    virtual int map_weights() override;
    virtual void run(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &d_y) override;
};

#endif /* RUNTIME_H */

