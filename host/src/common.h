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
#include "imagenet.h"
#include "lenet5.h"

#define CONCURRENT_EXECUTION

//#define OPENCL_PROFILER_ENABLE
//#define INTEL_PROFILER_ENABLE

extern int TEST_SET_SIZE;

extern cl_ulong kernel_time, write_time, read_time;

namespace config {

const char file_weight[] = "../data/mnist_weight_dump.txt";
//const Layer *cfg_network = ImageNet::SqueezeNet;  
const Layer *cfg_network = LeNet5::reuse_fused_network;

}

#endif /* COMMON_H */

