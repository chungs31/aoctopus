/* 2019-10-01 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * common.h
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include "CL/opencl.h"

//#define CONCURRENT_EXECUTION

//#define OPENCL_PROFILER_ENABLE
//#define INTEL_PROFILER_ENABLE

extern int TEST_SET_SIZE;

extern cl_ulong kernel_time, write_time, read_time;

namespace config {

char file_weight[] = "../data/mnist_weight_dump.txt";

}

#endif /* COMMON_H */

