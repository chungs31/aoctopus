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
#include <cstring>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "flags.h"

int buffer_mapper(int n_args, int for_input_idx);

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
    static int num_kernels; 
    static cl_command_queue write_queue;


    // stuff to keep const
    cl_device_id device;
    cl_program program;


    // CL Buffers
    aocl_utils::scoped_array<cl_mem> bufs;
    aocl_utils::scoped_array<size_t> buf_lens;
    int n_bufs;

    int output_idx;
    int input_idx;

    // Status flags
    bool weights_copied = false;
    bool is_input_layer = false;
    bool is_output_layer = false;
public:
    cl_ulong kernel_time = 0;
    static int num_copied;
    volatile static int num_ready;
    // Host memory
    aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > host_mems;
    std::vector<cl_mem_flags> buf_mflags;
    //aocl_utils::scoped_array<float> output;
    
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
    void copy_weights_to_bufs(bool use_positional_copy);
    static void wait_for_write_queue();

    // Set CL arguments
    void set_buffer_from_prev(const Octokernel *prev);

    // Enqueue
    void enqueue_kernel();
    void enqueue_kernel(int init);
    void enqueue_kernel_reuse();
    void enqueue_kernel_reuse(int init);

    // Input from vector or aocl_utils::scoped aligned ptr. 
    void set_input_mem(std::vector<float> &in) {
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
        //std::memcpy(host_mems[input_idx], in, buf_lens[input_idx] * sizeof(float));
    };
    
    void set_input_mem(aocl_utils::scoped_aligned_ptr<float> &in) {
        /*
        for (int i = 0; i < buf_lens[input_idx]; i++) {
            host_mems[input_idx][i] = in[i];
        }
        */
        std::memcpy(host_mems[input_idx], in, buf_lens[input_idx] * sizeof(float));
    };

    aocl_utils::scoped_aligned_ptr<float> &get_output_mem() {
        return host_mems[output_idx];
    };

    int get_output_idx() const { return output_idx; };
    int get_input_idx() const { return input_idx; };
    size_t get_buf_size(int idx) const { return buf_lens[idx]; }; 
    int get_n_bufs() const { return n_bufs; };

    void copy_output_from_to(aocl_utils::scoped_aligned_ptr<float> &out) {
        //out.reset(buf_lens[output_idx]);
        std::memcpy(out, host_mems[output_idx], buf_lens[output_idx] * sizeof(float));
    }
   
    /*
    void copy_output_from_to_fcn(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &out) {
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
    */

    void set_as_input_layer() { is_input_layer = true; };
    void set_as_output_layer() { is_output_layer = true; };

    bool is_input_or_output_layer() const{ return ( is_input_layer || is_output_layer ); };

    // debug functions
    void dbg_dump_output();
};
    
#endif

