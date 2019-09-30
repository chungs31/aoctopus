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

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"


using namespace aocl_utils;

class Octokernel {
private:
    // Kernels
    cl_kernel kernel;
    std::string kernel_name;

    // CL Buffers
    scoped_array<cl_mem> bufs;
    scoped_array<size_t> buf_lens;
    int n_bufs;

    std::vector<cl_mem_flags> buf_mflags;
    int output_idx;
    int input_idx;

    // Status flags
    bool weights_copied = false;
    
    // Events (for in-order execution)
    //scoped_array<cl_event> write_events;
public:
    // Host memory
    scoped_array<scoped_aligned_ptr<float> > host_mems;
    //scoped_array<float> output;
    
    // Public functions
    Octokernel(cl_context &context,
               cl_program &program, 
               const char *_kernel_name, 
               int num_buffers, 
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
    void copy_weights_to_bufs(cl_command_queue &q);

    // Set CL arguments
    void set_args();

    // Enqueue
    void enqueue_kernel(cl_command_queue &q);

    // Input from vector or scoped aligned ptr. 
    void set_input_mem(std::vector<float> &in) {
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
    };
    
    void set_input_mem(scoped_aligned_ptr<float> &in) {
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
    };

    scoped_aligned_ptr<float> &get_output_mem() {
        return host_mems[output_idx];
    };

    int get_output_idx() { return output_idx; };
    int get_input_idx() { return input_idx; };

    void copy_output_from_to(scoped_array<float> &out) {
        out.reset(buf_lens[output_idx]);

        for (int i = 0; i < buf_lens[output_idx]; i++) {
            out[i] = host_mems[output_idx][i];
        }
    }

    // debug functions
    void dbg_dump_output();
};

Octokernel::Octokernel(cl_context &context, cl_program &program, const char *_kernel_name, int num_buffers, std::vector<size_t> const &buffer_sizes, std::vector<cl_mem_flags> const &buffer_mflags, int output_idx, int input_idx) : 
    buf_mflags(buffer_mflags),
    output_idx(output_idx),
    input_idx(input_idx)
{
    // Store name of kernel
    kernel_name = _kernel_name;  

    // Initialize kernel
    cl_int status;
    kernel = clCreateKernel(program, _kernel_name, &status);
    checkError(status, "Failed to create kernel");
    
    n_bufs = num_buffers;
    host_mems.reset(n_bufs);
    buf_lens.reset(n_bufs);
    bufs.reset(n_bufs);
    
    for (int i = 0; i < n_bufs; i++) {
        // Initialize CL buffers
        buf_lens[i] = buffer_sizes[i];
        bufs[i] = clCreateBuffer(context, buffer_mflags[i], buf_lens[i] * sizeof(float) , NULL, &status);
        checkError(status, "Failed to create buffer for bias");
        //printf("Created buffer size %ld @ at %p with flag %lu\n", buf_lens[i], bufs[i], buffer_mflags[i]);

        // Initialize CPU variables
        host_mems[i].reset(buf_lens[i]);
    }

    //write_events.reset(4);
    //output.reset(120);
}

Octokernel::~Octokernel() {
    clReleaseKernel(kernel);
    for (int i = 0; i < n_bufs; i++) {
        clReleaseMemObject(bufs[i]);
    }

    //write_events.reset();
    host_mems.reset();
    buf_lens.reset();
    bufs.reset();
}

void Octokernel::load_buf(int buf_idx, std::vector<float> &in) {
    for (int i = 0; i < buf_lens[buf_idx]; i++) {
        host_mems[buf_idx][i] = in[i];
    }
}

void Octokernel::copy_weights_to_bufs(cl_command_queue &q) {
    /* Write kernel weights and biases to the device.
     * This should only be called once.
     */
    assert(weights_copied == false);

    cl_int status;

    std::vector<cl_event> write_events;

    for (int i = 0; i < n_bufs; i++) { // exclude the last buffer; this is the output
        if (buf_mflags[i] == CL_MEM_READ_ONLY && i != input_idx) {
            cl_event ev;
            printf("Copying buf %d with len %lu\n", i, buf_lens[i]);
            status = clEnqueueWriteBuffer(q, bufs[i], CL_FALSE, 0, buf_lens[i]* sizeof(float), host_mems[i], 0, NULL, &ev);
            checkError(status, "Failed to transfer to cl buf");
            write_events.push_back(ev);
        }
    }

    // Wait until weights and biases are transferred.
    cl_event *ev_ptr = &write_events[0];
    clWaitForEvents(write_events.size(), ev_ptr);

    for (auto &ev : write_events) {
        clReleaseEvent(ev);     
    }

    weights_copied = true;
}

void Octokernel::set_args() {
    cl_int status;
    
    for (unsigned i = 0; i < n_bufs; i++) {
        status = clSetKernelArg(kernel, i, sizeof(cl_mem), &bufs[i]);
        checkError(status, "Failed to set argument %d", i);
    }
}

void Octokernel::enqueue_kernel(cl_command_queue &q) {
    cl_int status;
    cl_event kernel_event, finish_event;
    cl_event write_event;

    //printf("Enqueue on %s\n", kernel_name.c_str());
    //printf("Number of buffers: %d\n", n_bufs);
    
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    status = clEnqueueWriteBuffer(q, bufs[input_idx], CL_FALSE, 0, buf_lens[input_idx]* sizeof(float), host_mems[input_idx], 0, NULL, &write_event);
    checkError(status, "Failed to transfer to cl buf");

    // set arguments
    for (unsigned i = 0; i < n_bufs; i++) {
        status = clSetKernelArg(kernel, i, sizeof(cl_mem), &bufs[i]);
        checkError(status, "Failed to set argument %d", i);
    }

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
    //printf("Launching kernel for  device %d (%zd elements)\n", i, global_work_size);
    //printf("Launching kernel\n");

    status = clEnqueueNDRangeKernel(q, kernel, 1, NULL,
            &global_work_size, NULL, 1, &write_event, &kernel_event); // change 3 to number of writes
    checkError(status, "Failed to launch kernel");
    
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(q, bufs[output_idx], CL_FALSE,
            0, buf_lens[output_idx]* sizeof(float), host_mems[output_idx], 1, &kernel_event, &finish_event);
    checkError(status, "Failed to launch kernel");
    
    clWaitForEvents(1, &finish_event);
    
    clFinish(q);
    clReleaseEvent(write_event);
    clReleaseEvent(finish_event);
    clReleaseEvent(kernel_event);
}

void Octokernel::dbg_dump_output() {
    std::string path = "activations/";
    path += kernel_name;
    path += ".txt";
    std::ofstream dumpfile(path.c_str());

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

#endif

