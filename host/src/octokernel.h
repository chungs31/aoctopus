/* 2019-09-18 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * octokernel.h
 *
 * Define the octokernel class. 
 *
 * Create cl_kernel objects and buffers related to the kernel (layer).
 *
 */
#ifndef OCTOKERNEL_H
#define OCTOKERNEL_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cstring>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "common.h"

cl_ulong write_time = 0, read_time = 0;

using namespace aocl_utils;

class Octokernel {
private:
    // Kernels
    int id;
    cl_kernel kernel;
    std::string kernel_name;
#ifdef CONCURRENT_EXECUTION
    cl_command_queue q = NULL;
#else
    static cl_command_queue q;
#endif


    // stuff to keep const
    cl_device_id device;
    cl_program program;


    static cl_command_queue write_queue;

    // CL Buffers
    scoped_array<cl_mem> bufs;
    scoped_array<size_t> buf_lens;
    int n_bufs;

    int output_idx;
    int input_idx;

    // Status flags
    bool weights_copied = false;
    bool is_input_layer = false;
    bool is_output_layer = false;
    bool inputs_copied = false;
    
    // Events (for in-order execution)
    static int num_kernels; 

    // For read-back

public:
    cl_ulong kernel_time = 0;
    static int num_copied;
    volatile static int num_ready;
    // Host memory
    scoped_array<scoped_aligned_ptr<float> > host_mems;
    std::vector<cl_mem_flags> buf_mflags;
    //scoped_array<float> output;
    
    // Public functions
    Octokernel(cl_context &context,
               cl_device_id &device,
               cl_program &program, 
               const char *_kernel_name, 
               //int num_buffers, 
               std::vector<size_t> const &buffer_sizes,
               std::vector<cl_mem_flags> const &buffer_mflags,
               int output_idx,
               int input_idx);
    ~Octokernel();

    // Copy weights from STL vectors into aligned pointers (host_mems).
    //void load_weights(std::vector<std::vector<float> > &weights);
    
    // Copy contents of in to respective host memory in host_mems.
    void load_buf(int buf_idx, std::vector<float> &in);

    void zero_buf(int buf_idx) {
        for (int i = 0; i < buf_lens[buf_idx]; i++) {
            host_mems[buf_idx][i] = 0.0;
        }
    };

    // Copy from host_mems to the CL buffers (bufs).
    void copy_weights_to_bufs();
    static void wait_for_write_queue();

    // Set CL arguments
    void set_buffer_from_prev(const Octokernel *prev);

    // Enqueue
    void enqueue_kernel();
    void enqueue_kernel(int init);
    void enqueue_kernel_reuse();
    void enqueue_kernel_reuse(int init);

    // Input from vector or scoped aligned ptr. 
    void set_input_mem(std::vector<float> &in) {
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
        //std::memcpy(host_mems[input_idx], in, buf_lens[input_idx] * sizeof(float));
    };
    
    void set_input_mem(scoped_aligned_ptr<float> &in) {
        /*
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
        */
        std::memcpy(host_mems[input_idx], in, buf_lens[input_idx] * sizeof(float));
    };

    scoped_aligned_ptr<float> &get_output_mem() {
        return host_mems[output_idx];
    };

    int get_output_idx() const { return output_idx; };
    int get_input_idx() const { return input_idx; };
    int get_n_bufs() const { return n_bufs; };

    void copy_output_from_to(scoped_aligned_ptr<float> &out) {
        //out.reset(buf_lens[output_idx]);
        std::memcpy(out, host_mems[output_idx], buf_lens[output_idx] * sizeof(float));
    }
    
    void copy_output_from_to_fcn(scoped_array<scoped_aligned_ptr<float> > &out) {
        while (num_copied < TEST_SET_SIZE) {
            if (num_ready > num_copied) {
                //for (int i = num_copied; i < num_ready; i++) {
                //std::cout << num_ready << ", " << num_copied << std::endl;
                std::memcpy(out[num_copied], host_mems[output_idx], buf_lens[output_idx] * sizeof(float));
                //}
                num_copied++;
            }
        }
    }

    void set_as_input_layer() { is_input_layer = true; };
    void set_as_output_layer() { is_output_layer = true; };

    // debug functions
    void dbg_dump_output();
};
    
#ifndef CONCURRENT_EXECUTION
cl_command_queue Octokernel::q = NULL;
#endif

cl_command_queue Octokernel::write_queue = NULL;

int Octokernel::num_kernels = 0;
int Octokernel::num_copied = 0;
volatile int Octokernel::num_ready = 0;

Octokernel::Octokernel(cl_context &context, cl_device_id &device, cl_program &program, const char *_kernel_name, std::vector<size_t> const &buffer_sizes, std::vector<cl_mem_flags> const &buffer_mflags, int output_idx, int input_idx) : 
    device(device),
    program(program),
    buf_mflags(buffer_mflags),

    output_idx(output_idx),
    input_idx(input_idx)
{
    // Store name of kernel
    kernel_name = _kernel_name;  

    // Initialize kernel
    id = num_kernels++;
    cl_int status;
    kernel = clCreateKernel(program, _kernel_name, &status);
    checkError(status, "Failed to create kernel");
    
    //n_bufs = num_buffers;
    //cl_uint num_args;
    status = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &n_bufs, NULL);
    //n_bufs;// minus 1 if using globalmem for init
    if (kernel_name == "write_to_ch7") {
        n_bufs--;
    }
    checkError(status, "Failed to get kernel num args");

    host_mems.reset(n_bufs);
    buf_lens.reset(n_bufs);
    bufs.reset(n_bufs);
    
    for (int i = 0; i < n_bufs; i++) {
        // Initialize CL buffers
        buf_lens[i] = buffer_sizes[i];
        bufs[i] = clCreateBuffer(context, buffer_mflags[i], buf_lens[i] * sizeof(float) , NULL, &status);
        checkError(status, "Failed to create buffer for bias");

        // Initialize CPU variables
        host_mems[i].reset(buf_lens[i]);
    }

    // This queue is for weight/bias buffer copying only.
    if (!write_queue) {
#ifdef OPENCL_PROFILER_ENABLE
        write_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
#else
        write_queue = clCreateCommandQueue(context, device, 0, &status);
#endif
    }
    checkError(status, "Failed to create command queue");

    // Queue for kernel.
    if (!q) {
#ifdef OPENCL_PROFILER_ENABLE
        q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
#else
        q = clCreateCommandQueue(context, device, 0, &status);
#endif
    }
    checkError(status, "Failed to create command queue");
}

Octokernel::~Octokernel() {
    if (kernel) 
        clReleaseKernel(kernel);
    if (q) 
        clReleaseCommandQueue(q);
    if (write_queue)
        clReleaseCommandQueue(write_queue);

    for (int i = 0; i < n_bufs; i++) {
        host_mems[i].reset();
        clReleaseMemObject(bufs[i]);
    }
    host_mems.reset();
    buf_lens.reset();
    bufs.reset();
}

void Octokernel::load_buf(int buf_idx, std::vector<float> &in) {
    for (int i = 0; i < buf_lens[buf_idx]; i++) {
        host_mems[buf_idx][i] = in[i];
    }
}

void Octokernel::copy_weights_to_bufs() {
    /* Write kernel weights and biases to the device.
     * This should only be called once.
     */
    assert(weights_copied == false);

    cl_int status;

    for (int i = 0; i < n_bufs; i++) { // exclude the last buffer; this is the output
        if (buf_mflags[i] == CL_MEM_READ_ONLY) {
            printf("Copying buf %d with len %lu\n", i, buf_lens[i]);
            status = clEnqueueWriteBuffer(write_queue, bufs[i], CL_FALSE, 0, buf_lens[i] * sizeof(float), host_mems[i], 0, NULL, NULL);
            checkError(status, "Failed to transfer to cl buf");
        }
    }

    // This is non-blocking! Call wait after this
    //enqueue_kernel(); // Load local caches. Only when necessary.
    weights_copied = true;
}

void Octokernel::wait_for_write_queue() {
    clFinish(write_queue);
}

void Octokernel::set_buffer_from_prev(const Octokernel *prev) {
    if (input_idx >= 0 && prev->output_idx >= 0) {
        bufs[input_idx] = prev->bufs[prev->output_idx];
    }
}

void Octokernel::enqueue_kernel() {
    cl_int status;
    cl_event kernel_event = NULL, finish_event = NULL, write_event = NULL;

    //printf("Enqueue on %s\n", kernel_name.c_str());
    //printf("Number of buffers: %d\n",n_bufs);
    
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    if (input_idx >= 0 && is_input_layer && !inputs_copied) {
        status = clEnqueueWriteBuffer(q, bufs[input_idx], CL_FALSE, 0, buf_lens[input_idx]* sizeof(float), host_mems[input_idx], 0, NULL, &write_event);
        checkError(status, "Failed to transfer to cl buf");
        //inputs_copied = true;
    }

    int tmp = 1;
    // set arguments
    for (unsigned i = 0; i < n_bufs; i++) {
        status = clSetKernelArg(kernel, i, sizeof(cl_mem), &bufs[i]);
        checkError(status, "Failed to set argument %d", i);
    }
    /*
    if (id != num_kernels - 1) {
        if (weights_copied) {
            status = clSetKernelArg(kernel, n_bufs, sizeof(int), &tmp);
        }
        else {
            tmp = 0;
            status = clSetKernelArg(kernel, n_bufs, sizeof(int), &tmp);
        }
    }
    */

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = 1;

    if (write_event) {
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 1, &write_event, &kernel_event); // change 3 to number of writes
    checkError(status, "Failed to launch kernel %s", kernel_name.c_str());
    }
    else {
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 0, NULL, &kernel_event); // change 3 to number of writes
    checkError(status, "Failed to launch kernel %s", kernel_name.c_str());
    }
    
    // Read the result. This the final operation.
    if (output_idx >= 0 && is_output_layer) {
    status = clEnqueueReadBuffer(q, bufs[output_idx], CL_TRUE,
            0, buf_lens[output_idx]* sizeof(float), host_mems[output_idx], 1, &kernel_event, &finish_event);
    checkError(status, "Failed to launch kernel");
    num_ready++;
    }
   
#ifdef OPENCL_PROFILER_ENABLE
    cl_ulong start, end;
    clFinish(q);

    //clGetProfileDataDeviceIntelFPGA(device, program, true, true, (cl_bool) NULL, (size_t) NULL, (void *) NULL, (size_t) NULL, (cl_int *) NULL);

    if (write_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(write_event);
#endif
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        write_time += end - start;
    }

    if (kernel_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(kernel_event);
#endif
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        kernel_time += end - start;
    }
    
    if (finish_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(finish_event);
#endif
        clGetEventProfilingInfo(finish_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(finish_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        read_time += end - start;
    }
#endif

    if (write_event) clReleaseEvent(write_event);
    if (kernel_event) clReleaseEvent(kernel_event);
    if (finish_event) clReleaseEvent(finish_event);
}
void Octokernel::enqueue_kernel_reuse() {
    enqueue_kernel_reuse(0);
}

void Octokernel::enqueue_kernel_reuse(int init) {
    cl_int status;
    cl_event k1, k2, k3;

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = 1;

    if (init == 1){
        int argid = 0;
        clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[0]);
        clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[1]);
        clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[2]);
        clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[7]); // first output
        int ax1_bound = 1;
        clSetKernelArg(kernel, argid++, sizeof(int), &ax1_bound);
        int k_bound = 1; 
        clSetKernelArg(kernel, argid++, sizeof(int), &k_bound);
        int relu_true = 0;
        clSetKernelArg(kernel, argid++, sizeof(int), &relu_true);
        int wait_channel = 0;
        clSetKernelArg(kernel, argid++, sizeof(int), &wait_channel);
        int done = 0;
        clSetKernelArg(kernel, argid++, sizeof(int), &done);
        clSetKernelArg(kernel, argid++, sizeof(int), &init);

        status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
                &global_work_size, NULL, 0, NULL, &k1); // change 3 to number of writes
        checkError(status, "Failed to launch kernel %s p1", kernel_name.c_str());
        clReleaseEvent(k1);
        return;
    }
        
    int argid = 0;
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[0]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[1]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[2]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[7]); // first output
    int ax1_bound = 120;
    clSetKernelArg(kernel, argid++, sizeof(int), &ax1_bound);
    int k_bound = 400; 
    clSetKernelArg(kernel, argid++, sizeof(int), &k_bound);
    int relu_true = 1;
    clSetKernelArg(kernel, argid++, sizeof(int), &relu_true);
    int wait_channel = 1;
    clSetKernelArg(kernel, argid++, sizeof(int), &wait_channel);
    int done = 0;
    clSetKernelArg(kernel, argid++, sizeof(int), &done);
    clSetKernelArg(kernel, argid++, sizeof(int), &init);

    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 0, NULL, &k1); // change 3 to number of writes
    checkError(status, "Failed to launch kernel %s p1", kernel_name.c_str());
    
    argid = 0;
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[7]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[3]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[4]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[8]); // second output
    ax1_bound = 84;
    clSetKernelArg(kernel, argid++, sizeof(int), &ax1_bound);
    k_bound = 120; 
    clSetKernelArg(kernel, argid++, sizeof(int), &k_bound);
    relu_true = 1;
    clSetKernelArg(kernel, argid++, sizeof(int), &relu_true);
    wait_channel = 0;
    clSetKernelArg(kernel, argid++, sizeof(int), &wait_channel);
    done = 0;
    clSetKernelArg(kernel, argid++, sizeof(int), &done);
    clSetKernelArg(kernel, argid++, sizeof(int), &init);
    
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 1, &k1, &k2); // k1 must finish before this
    checkError(status, "Failed to launch kernel %s p2", kernel_name.c_str());
    
    argid = 0;
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[8]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[5]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[6]);
    clSetKernelArg(kernel, argid++, sizeof(cl_mem), &bufs[9]); // third output
    ax1_bound = 10;
    clSetKernelArg(kernel, argid++, sizeof(int), &ax1_bound);
    k_bound = 84; 
    clSetKernelArg(kernel, argid++, sizeof(int), &k_bound);
    relu_true = 0;
    clSetKernelArg(kernel, argid++, sizeof(int), &relu_true);
    wait_channel = 0;
    clSetKernelArg(kernel, argid++, sizeof(int), &wait_channel);
    done = 1;
    clSetKernelArg(kernel, argid++, sizeof(int), &done);
    clSetKernelArg(kernel, argid++, sizeof(int), &init);
    
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 1, &k2, &k3); // k1 must finish before this
    checkError(status, "Failed to launch kernel %s p3", kernel_name.c_str());
    
    // Read the result. This the final operation.
    /*if (output_idx >= 0 && is_output_layer) {
    status = clEnqueueReadBuffer(q, bufs[output_idx], CL_TRUE,
            0, buf_lens[output_idx]* sizeof(float), host_mems[output_idx], 1, &k3, &finish_event);
    checkError(status, "Failed to launch kernel");
    num_ready++;
    }
    */
#ifdef OPENCL_PROFILER_ENABLE
    cl_ulong start, end;
    clFinish(q);


#ifdef INTEL_PROFILER_ENABLE
    clGetProfileInfoIntelFPGA(k1);
    clGetProfileInfoIntelFPGA(k2);
    clGetProfileInfoIntelFPGA(k3);
#endif
    clGetEventProfilingInfo(k1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(k1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    kernel_time += end - start;
    clGetEventProfilingInfo(k2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(k2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    kernel_time += end - start;
    clGetEventProfilingInfo(k3, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(k3, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    kernel_time += end - start;
#endif
   
    clReleaseEvent(k1);
    clReleaseEvent(k2);
    clReleaseEvent(k3);
    //return kernel_event;
}

void Octokernel::dbg_dump_output() {
    std::string path = "activations/";
    path += kernel_name;
    path += ".txt";
    std::ofstream dumpfile(path.c_str());
    
    cl_int status;
    status = clEnqueueReadBuffer(q, bufs[output_idx], CL_TRUE,
            0, buf_lens[output_idx]* sizeof(float), host_mems[output_idx], 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    dumpfile << kernel_name << std::endl;
    dumpfile << buf_lens[output_idx] << std::endl << "[";
    for (int i = 0, counter = 0; i < buf_lens[output_idx]; i++, counter++) {
        if (i != buf_lens[output_idx] - 1) {
            //printf("%f, ", host_mems[output_idx][i]);
            dumpfile << host_mems[output_idx][i] << ", ";
        }
        else {
            //printf("%f", host_mems[output_idx][i]);
            dumpfile << host_mems[output_idx][i];
        }
        if (counter == 3) {
            dumpfile << std::endl;
            counter = 0;
        }
    }
    //printf("]\n");
    dumpfile << "]" << std::endl;

    dumpfile.close();
}

void Octokernel::enqueue_kernel(int init) {
    cl_int status;
    cl_event kernel_event = NULL, finish_event = NULL, write_event = NULL;

    //printf("Enqueue on %s\n", kernel_name.c_str());
    //printf("Number of buffers: %d\n",n_bufs);
    
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    if (input_idx >= 0 && is_input_layer && !inputs_copied) {
        status = clEnqueueWriteBuffer(q, bufs[input_idx], CL_FALSE, 0, buf_lens[input_idx]* sizeof(float), host_mems[input_idx], 0, NULL, &write_event);
        checkError(status, "Failed to transfer to cl buf");
    }

    int tmp = 1;
    // set arguments
    for (unsigned i = 0; i < n_bufs; i++) {
        status = clSetKernelArg(kernel, i, sizeof(cl_mem), &bufs[i]);
        checkError(status, "Failed to set argument %d", i);
    }
    clSetKernelArg(kernel, n_bufs, sizeof(int), &init);

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = 1;

    if (write_event) {
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 1, &write_event, &kernel_event); // change 3 to number of writes
    checkError(status, "Failed to launch kernel %s", kernel_name.c_str());
    }
    else {
    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 0, NULL, &kernel_event); // change 3 to number of writes
    checkError(status, "Failed to launch kernel %s", kernel_name.c_str());
    }
    
    // Read the result. This the final operation.
    if (output_idx >= 0 && is_output_layer) {
    status = clEnqueueReadBuffer(q, bufs[output_idx], CL_TRUE,
            0, buf_lens[output_idx]* sizeof(float), host_mems[output_idx], 1, &kernel_event, &finish_event);
    checkError(status, "Failed to launch kernel");
    num_ready++;
    }
   
#ifdef OPENCL_PROFILER_ENABLE
    cl_ulong start, end;
    clFinish(q);

    //clGetProfileDataDeviceIntelFPGA(device, program, true, true, (cl_bool) NULL, (size_t) NULL, (void *) NULL, (size_t) NULL, (cl_int *) NULL);

    if (write_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(write_event);
#endif
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        write_time += end - start;
    }

    if (kernel_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(kernel_event);
#endif
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        kernel_time += end - start;
    }
    
    if (finish_event) {
#ifdef INTEL_PROFILER_ENABLE
        clGetProfileInfoIntelFPGA(finish_event);
#endif
        clGetEventProfilingInfo(finish_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(finish_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        read_time += end - start;
    }
#endif

    if (write_event) clReleaseEvent(write_event);
    if (kernel_event) clReleaseEvent(kernel_event);
    if (finish_event) clReleaseEvent(finish_event);
    //return kernel_event;
}
    
#endif

