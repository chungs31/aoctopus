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

//#define CONCURRENT_EXECUTION

//#define OPENCL_PROFILER_ENABLE
//#define INTEL_PROFILER_ENABLE

extern int TEST_SET_SIZE;

extern cl_ulong kernel_time, write_time, read_time;

namespace config {

//const char file_weight[] = "../data/inet_sqnet_params.txt";
//const char file_weight[] = "../data/mnist_weight_dump.txt";
const char file_weight[] = "../data/inet_mnet_params.txt";

const Layer *cfg_network = ImageNet::MobileNet;  
//const Layer *cfg_network = LeNet5::autorun_network;

//const char file_bufsizes[] = "../data/inet_sqnet_bufsizes.txt";
const char file_bufsizes[] = "../data/inet_mnet_bufsizes.txt";

const char input_image[] = "../data/cat224224.db";

}

#endif /* COMMON_H */

