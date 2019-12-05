#include <vector>
#include <iostream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "test_pcie_bandwidth.h"

using namespace aocl_utils;

cl_platform_id platform = NULL;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
unsigned num_devices = 0;

float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    // Create the context.
    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    return true;
}

void pcie_bandwidth_test() {
    std::vector<int> sizes = {8,16,32,64,128,256,512,1024,2048,4096,8192,16834,32768,65536,131072,
        262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432};
    //std::vector<cl_mem_flags> flags = {CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY}; 
    std::vector<cl_mem_flags> flags = {CL_MEM_READ_WRITE}; 
    cl_int status;
        
    cl_command_queue q = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create cq");

    printf("MEM_FLAG, n_elements, bytes, write time, copy time, read time, write speed, copy speed, read speed\n");
    
    for (auto &memflag : flags) {
        for (auto &size : sizes) {
            cl_mem d_buf = clCreateBuffer(context, memflag, size * sizeof(float), NULL, &status);
            checkError(status, "Failed to create test buffer");
            cl_mem d_copy_buf = clCreateBuffer(context, memflag, size * sizeof(float), NULL, &status);
            checkError(status, "Failed to create test buffer");
            
            cl_ulong write_time, copy_time, read_time;
            cl_ulong start, end;

            scoped_aligned_ptr<float> h_buf, h_copy_buf;
            h_buf.reset(size);
            h_copy_buf.reset(size);

            for (int i = 0; i < size; i++) {
                h_buf[i] = rand_float();
            }

            cl_event read_event, copy_event, write_event;

            for (int j = 0; j < 100; j++) {
                status = clEnqueueWriteBuffer(q, d_buf, CL_TRUE, 0, size * sizeof(float), h_buf, 0, NULL, &write_event);
                checkError(status, "Failed to write");
                status = clEnqueueCopyBuffer(q, d_buf, d_copy_buf, 0, 0, size * sizeof(float), 1, &write_event, &copy_event);
                checkError(status, "Failed to copy");
                status = clEnqueueReadBuffer(q, d_copy_buf, CL_TRUE, 0, size * sizeof(float), h_copy_buf, 1, &copy_event, &read_event);
                checkError(status, "Failed to read");

                clFinish(q);

                status = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                checkError(status, "Failed to get things");
                status = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                write_time += end - start;
                status = clGetEventProfilingInfo(copy_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                status = clGetEventProfilingInfo(copy_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                copy_time += end - start;
                status = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                status = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                read_time += end - start;
                
                clReleaseEvent(read_event);
                clReleaseEvent(copy_event);
                clReleaseEvent(write_event);
            }

            write_time /= 100;
            copy_time /= 100;
            read_time /= 100;


            double numbytes = (double) (size * sizeof(float));
            printf("%d, %8d, %8d, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n", 
                    (int) memflag, (int) size, (int) numbytes, 
                    ((double) write_time) / 1000000, // ms
                    ((double) copy_time) / 1000000, 
                    ((double) read_time) / 1000000,
                    1000 * (double)(numbytes / (double) write_time), // MB/s
                    1000 * (double)(numbytes / (double) copy_time), 
                    1000 * (double)(numbytes / (double) read_time));


            clReleaseMemObject(d_buf);
            clReleaseMemObject(d_copy_buf);
            h_buf.reset();
            h_copy_buf.reset();
        }
    }

    clReleaseCommandQueue(q);
}

void cleanup() {
    if (context) {
        clReleaseContext(context);
    }
}

int main() {
    if(!init_opencl()) {
        return -1;
    }

    pcie_bandwidth_test();

    cleanup();

    return 0;
}

