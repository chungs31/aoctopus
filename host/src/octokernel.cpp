#include <assert.h>
#include <iostream>
#include <fstream>

#include "octokernel.h"

cl_ulong write_time = 0, read_time = 0;
using namespace aocl_utils;

#ifndef CONCURRENT_EXECUTION
cl_command_queue Octokernel::q = NULL;
#endif

cl_command_queue Octokernel::write_queue = NULL;

int Octokernel::num_kernels = 0;
int Octokernel::num_copied = 0;
volatile int Octokernel::num_ready = 0;

Octokernel::Octokernel(cl_context &context, cl_device_id &device, cl_program &program, const char *_kernel_name, std::vector<size_t> const &buffer_sizes, std::vector<cl_mem_flags> const &buffer_mflags, int _output_idx, int _input_idx) : 
    device(device),
    program(program),
    buf_mflags(buffer_mflags),

    output_idx(_output_idx),
    input_idx(_input_idx)
{
    // Store name of kernel
    kernel_name = _kernel_name;  

    // Initialize kernel
    id = num_kernels++;
    cl_int status;
    kernel = clCreateKernel(program, _kernel_name, &status);
    checkError(status, "Failed to create kernel");
    
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


    /*
    if (input_idx < 0) {
        input_idx = buffer_mapper(n_bufs, 1);   
    }
    if (output_idx < 0) {
        output_idx = buffer_mapper(n_bufs, 0);
    }
    */

    std::cout << "[DEBUG] kernel " << id << " w name " << kernel_name << ", in: " << input_idx << ", out: " << output_idx << "\n";
    
    for (int i = 0; i < n_bufs; i++) {
        // Initialize CL buffers
        if (!buffer_sizes.empty())
            buf_lens[i] = buffer_sizes[i];
        
        if (buffer_mflags.empty()) {
            bufs[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_lens[i] * sizeof(float) , NULL, &status);
        }
        else {
            bufs[i] = clCreateBuffer(context, buffer_mflags[i], buf_lens[i] * sizeof(float) , NULL, &status);
        }

        std::cout << "acquiring: " << buf_lens[i] * sizeof(float) << std::endl;
        checkError(status, "Failed to create buffer for bias");

        // Initialize CPU variables
        //if (i == 2 || i == 4 || i == 5) {
          host_mems[i].reset(buf_lens[i]);
        //}
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
        //clReleaseMemObject(bufs[i]);
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
        //if ((n_bufs == 5 && (i == 2 || i == 4)) || (n_bufs == 6 && (i == 2 || i == 4 || i == 5))) {
        //if (buf_mflags[i] == CL_MEM_READ_ONLY) {
            printf("Copying buf %d with len %lu\n", i, buf_lens[i]);
            status = clEnqueueWriteBuffer(write_queue, bufs[i], CL_FALSE, 0, buf_lens[i] * sizeof(float), host_mems[i], 0, NULL, NULL);
            checkError(status, "Failed to transfer to cl buf");
        }
    }

    // This is non-blocking! Call wait after this
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

int buffer_mapper(int n_args, int for_input_idx) {
    int ret = 0;

    if (for_input_idx == 1) { // request is for input_idx
        switch (n_args) {
            default:
                ret = 1;
                break;
        }
    }
    else { // request is for output_idx
        switch (n_args) {
            case 2:
            case 3:
                ret = 0;
                break;
            case 4:
            case 5:
            case 6:
                ret = 3;
                break;
        }
    }

    return ret;
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

    std::cout << "[DEBUG] Dumping output for kernel " << kernel_name << " with output_idx " << output_idx << std::endl;
    
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
    
