
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float ch0 __attribute__((depth(4056)));
channel float ch1 __attribute__((depth(1014)));
channel float ch2 __attribute__((depth(1936)));
channel float ch3 __attribute__((depth(400)));
channel float ch4 __attribute__((depth(400)));
channel float ch5 __attribute__((depth(120)));
channel float ch6 __attribute__((depth(84)));
channel float ch7 __attribute__((depth(10)));

channel int reader_valid __attribute__((depth(1)));

channel int dense_ready __attribute__((depth(1)));
channel int dense_valid __attribute__((depth(1)));

channel int writer_ready __attribute__((depth(1)));

__kernel void fuse_conv2d_relu_kernel0(
    __global const float* restrict input0, 
    __global const float* restrict input1, 
    __global const float* restrict input2) 
{
  float lcompute[676];
  //float out[4056];
  float input0_cache[784];

  for (int yy = 0; yy < 28; ++yy) {
    for (int xx = 0; xx < 28; ++xx) {
      input0_cache[((yy * 28) + xx)] = input0[((yy * 28) + xx)];
    }
  }

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 26; ++yy) {
      for (int xx = 0; xx < 26; ++xx) {
        lcompute[((yy * 26) + xx)] = 0.000000e+00f;
        #pragma unroll
        for (int ry = 0; ry < 3; ++ry) {
          #pragma unroll
          for (int rx = 0; rx < 3; ++rx) {
            lcompute[((yy * 26) + xx)] += input0_cache[((((yy + ry) * 28) + xx) + rx)] * input1[((((ax1 * 3) + ry) * 3) + rx)];
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      for (int ax3 = 0; ax3 < 26; ++ax3) {
        //out[((((ax1 * 26) + ax2) * 26) + ax3)] = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
        float tmp = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
        write_channel_intel(ch0, tmp);
      }
    }
  }

  //for (int i = 0; i < 4056; ++i) {
  //  write_channel_intel(ch0, out[i]);
  // }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fuse_avg_pool2d_kernel0() {
  float ltensor[1014];
  float in[4056];

  for (int i = 0; i < 4056; ++i) {
      in[i] = read_channel_intel(ch0);      
  }

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 13; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 13; ++ax3) {
        ltensor[((((ax1 * 13) + ax2) * 13) + ax3)] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            ltensor[((((ax1 * 13) + ax2) * 13) + ax3)] = (ltensor[((((ax1 * 13) + ax2) * 13) + ax3)] + (in[((((((((ax1 * 13) + ax2) * 2) + rv) * 13) + ax3) * 2) + rv1)] * 2.500000e-01f));
          }
        }
      }
    }
  }

  for (int i = 0; i < 1014; ++i) {
    write_channel_intel(ch1, ltensor[i]);
  }
}

__kernel void fuse_conv2d_relu_1_kernel0(
    __global const float* restrict input1, 
    __global const float* restrict input2
) {
  float lcompute[121];
  float in[1014];
  float out[1936];
  
  for (int i = 0; i < 1014; ++i) {
      in[i] = read_channel_intel(ch1);      
  }
  
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int yy = 0; yy < 11; ++yy) {
      for (int xx = 0; xx < 11; ++xx) {
        lcompute[((yy * 11) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 6; ++rc) {
          #pragma unroll
          for (int ry = 0; ry < 3; ++ry) {
            #pragma unroll
            for (int rx = 0; rx < 3; ++rx) {
              lcompute[((yy * 11) + xx)] = (lcompute[((yy * 11) + xx)] + (in[((((((rc * 13) + yy) + ry) * 13) + xx) + rx)] * input1[((((((ax1 * 6) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 11; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 11; ++ax3) {
        out[((((ax1 * 11) + ax2) * 11) + ax3)] = max((lcompute[((ax2 * 11) + ax3)] + input2[ax1]), 0.000000e+00f);
      }
    }
  }
  
  for (int i = 0; i < 1924; ++i) {
    write_channel_intel(ch2, out[i]);
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fuse_avg_pool2d_1_kernel0() {
  float ltensor[400];
  float in[1936];

  for (int i = 0; i < 1924; ++i) {
      in[i] = read_channel_intel(ch2);      
  }

  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        ltensor[((((ax1 * 5) + ax2) * 5) + ax3)] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            ltensor[((((ax1 * 5) + ax2) * 5) + ax3)] = (ltensor[((((ax1 * 5) + ax2) * 5) + ax3)] + (in[(((((ax1 * 121) + (ax2 * 22)) + (rv * 11)) + (ax3 * 2)) + rv1)] * 2.500000e-01f));
          }
        }
      }
    }
  }
  for (int i = 0; i < 400; ++i) {
    write_channel_intel(ch3, ltensor[i]);
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fuse_transpose_flatten_kernel0() {
  float in[400];
  for (int i = 0; i < 400; ++i) {
    in[i] = read_channel_intel(ch3);
  }

  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 400; ++ax0_ax1_fused_inner) {
    //tensor[ax0_ax1_fused_inner] = input0[(((ax0_ax1_fused_inner % 16) * 25) + (ax0_ax1_fused_inner / 16))];
    write_channel_intel(ch4, in[(((ax0_ax1_fused_inner % 16) * 25) + (ax0_ax1_fused_inner / 16))]);
  }
}


/* generic templates */ 

kernel void fuse_dense_reuse(
    global float* restrict in,
    global const float* restrict weight, 
    global const float* restrict bias,
    global float* restrict out,
    int ax1_bound,
    int k_bound,
    int relu_true,
    int wait_channel, // should be 1 on the first exec, 0 for else.
    int done,// if its done, tell next kernel to write to channel
    int init
) {
    if (wait_channel) {
        // First stage -- need to know if input is valid
        for (int i = 0; i < 400; i++) {
            in[i] = read_channel_intel(ch4); // Write to gmem
        }
    }

    for (int ax1 = 0; ax1 < ax1_bound; ax1++) {
        float sum = 0.00e+00f;

        #pragma unroll 10
        for (int k = 0; k < k_bound; k++) {
            sum += in[k] * weight[((ax1 * k_bound) + k)];
        }
        float tmp = sum + bias[ax1];
        if (relu_true) {
            tmp = max(tmp, 0.00e+00f);
        }
        out[ax1] = tmp;
    }

    if (done) {
      for (int i = 0; i < 10; i++) {
        write_channel_intel(ch7, out[i]);
      }
    }
}

__kernel void fuse_softmax_kernel0(
    __global float* restrict tensor2
) {
  float in[10];
  for (int i = 0; i < 10; i++) {
      in[i] = read_channel_intel(ch7);      
  }

  #pragma unroll
  for (int ax1 = 0; ax1 < 10; ++ax1) {
    float tmp0 = -3.402823e+38f;
    #pragma unroll
    for (int k1 = 0; k1 < 10; ++k1) {
      tmp0 = max(tmp0, in[k1]);
    }
    float tmp1 = 0.000000e+00f;
    #pragma unroll
    for (int k2 = 0; k2 < 10; ++k2) {
      tmp1 = (tmp1 + exp((in[k2] - tmp0)));
    }
    tensor2[ax1] = (exp((in[ax1] - tmp0)) / tmp1);
  }
}
