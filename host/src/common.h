/* 2019-10-01 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * common.h
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include "CL/opencl.h"
#include "importer.h"
#include "runtime.h"
#include <string>

class Layer;

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

extern OctoCfg LeNet5;
extern OctoCfg MobileNetV2;
extern OctoCfg SqueezeNet;

extern OctoCfg *octocfg;

}

#endif /* COMMON_H */

