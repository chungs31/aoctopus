/* 2019-10-01 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * common.h
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include "CL/opencl.h"
#include "layer.h"
//#include "imagenet.h"
#include "networks.h"
#include "importer.h"
#include "runtime.h"
#include <string>

/* Concurrent Execution
 *
 * Controls the creation of the command queue for kernels. If it is defined, every Octokernel 
 * object will have its own private command queue, allowing for concurrent execution of
 * queued tasks. If undefined, the command queue will be declared static to the class, effectively
 * creating a single in-order command queue that serializes execution.
 */
#define CONCURRENT_EXECUTION

/* OpenCL Profiler
 *
 * When defined, event profilers will be enabled to capture execution/read/write times. Concurrent
 * execution must be undefined/disabled.
 */
//#define OPENCL_PROFILER_ENABLE

/* Intel Profiler
 *
 * When defined, event profilers will be sent to the Intel profiler to generate an execution trace
 * (profile.mon). Use aocl report to analyze the output. OpenCL Profiler must be enabled.
 */
//#define INTEL_PROFILER_ENABLE

/* External shared variables
 */
extern int TEST_SET_SIZE;
extern cl_ulong kernel_time, write_time, read_time;

/* Configuration
 */
namespace config {

struct OctoCfg {
    std::string f_weight;
    std::string f_bufsizes;
    Layer *cfg_network;
    Importer importer;
    Executor executor;
};

OctoCfg LeNet5 {
    .f_weight="../data/mnist_weight_dump.txt",
    .f_bufsizes="",
    .cfg_network = LeNet5::reuse_fused_network,
    .importer = Importer(10000, 784, "../data/mnist_test.db", "../data/mnist_test_y.db"),
    .executor = Executor(10000, 10)
};

OctoCfg MobileNetV2 {
    .f_weight="../data/inet_mnet_params.txt",
    .f_bufsizes="../data/inet_mnet_channels_bufsizes.txt",
    .cfg_network = ImageNet::MobileNet_channels,
    .importer = Importer(1, 224*224*3, "../data/cat244244.db", ""), 
    .executor = Executor(1, 1000)
};

OctoCfg SqueezeNet {
    .f_weight="../data/inet_sqnet_params.txt",
    .f_bufsizes="",
    .cfg_network = ImageNet::SqueezeNet,
    .importer = Importer(1, 224*224*3, "../data/cat244244.db", ""),
    .executor = Executor(1, 1000)
};

/* Modify this to choose network */
OctoCfg *octocfg = &LeNet5;

}

#endif /* COMMON_H */

