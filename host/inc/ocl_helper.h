/* 2020-02-10 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * ocl_helper.h
 *
 * Helper functions for OpenCL structures and enumerations.
 */

#ifndef OCL_HELPER_H
#define OCL_HELPER_H

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

// OpenCL runtime configuration
struct ocl_info {
  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_program program = NULL;
  aocl_utils::scoped_array<cl_device_id> device;
  unsigned num_devices = 0;
};

extern struct ocl_info oclinfo;
extern bool use_fast_emulator;
extern int num_kernels;

// Devices
extern unsigned num_devices;

// Kernels
const cl_uint max_kernels_supported = 128;
extern cl_kernel kernels[max_kernels_supported];

// Debug info
extern double wall_clock_time;

// Function prototypes
bool init_opencl(const std::string);
bool init_opencl_internals(const std::string);

void init_problem();
bool run();
void cleanup();
void profiler_output();

#endif /* OCL_HELPER_H */
