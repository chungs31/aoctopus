#include <cstdlib>
#include "ocl_helper.h"

using namespace aocl_utils;

// OpenCL runtime configuration
ocl_info oclinfo;
int num_kernels;
cl_kernel kernels[max_kernels_supported];
bool use_fast_emulator = false;

bool init_opencl_internals(const std::string f_bitstream) {
  cl_int status;

  if(!setCwdToExeDir()) {
      return false;
  }

  // Get the OpenCL oclinfo.platform.
  if (use_fast_emulator) {
      oclinfo.platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
  } else {
      oclinfo.platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  }
  if(oclinfo.platform == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
  }

  // Query the available OpenCL oclinfo.device.
  oclinfo.device.reset(getDevices(oclinfo.platform, CL_DEVICE_TYPE_ALL, &oclinfo.num_devices));
  printf("Platform: %s\n", getPlatformName(oclinfo.platform).c_str());
  printf("Using %d oclinfo.device(s)\n", oclinfo.num_devices);
  for(unsigned i = 0; i < oclinfo.num_devices; ++i) {
      printf("  %s\n", getDeviceName(oclinfo.device[i]).c_str());
  }

  // Create the oclinfo.context.
  oclinfo.context = clCreateContext(NULL, oclinfo.num_devices, oclinfo.device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the oclinfo.program for all oclinfo.device. Use the first oclinfo.device as the
  // representative oclinfo.device (assuming all oclinfo.device are of the same type).
  std::string path_home(getenv("HOME"));
  std::string binary_file = getBoardBinaryFile((path_home + "/bitstreams/" + f_bitstream).c_str(), oclinfo.device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  oclinfo.program = createProgramFromBinary(oclinfo.context, binary_file.c_str(), oclinfo.device, oclinfo.num_devices);

  // Build the oclinfo.program that was just created.
  status = clBuildProgram(oclinfo.program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");
}

// Free the resources allocated during initialization
void cleanup() {
    for (int i = 0; i < num_kernels; i++) {
        clReleaseKernel(kernels[i]);
    }

    if (oclinfo.program) {
        clReleaseProgram(oclinfo.program);
    }
    if (oclinfo.context) {
        clReleaseContext(oclinfo.context);
    }
}

// Print output for profiler in seconds.
void profiler_output() {
#ifdef OPENCL_PROFILER_ENABLE
    printf("OpenCL event profiler output\n");
    double sum = 0;
    for (int kernel = 0; kernel < num_kernels; kernel++) {
        sum += octokernels[kernel]->kernel_time;
        printf("Kernel %d execution time: %f ms\n", kernel, (double) octokernels[kernel]->kernel_time / 1000000.0);
    }
    printf("Total Kernel execution time: %f ms\n", (double) sum / 1000000.0);
    printf("Write to FPGA time: %f ms\n", (double) write_time / 1000000.0);
    printf("Read to FPGA time: %f ms\n", (double) read_time / 1000000.0);
    printf("Idle time: %f ms\n", wall_clock_time - (double)(sum+write_time+read_time)/1000000.0);
#endif
    printf("Wall clock time: %0.3f ms\n", wall_clock_time);
}
