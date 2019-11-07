
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float ch0 __attribute__((depth(4056)));
channel float ch1 __attribute__((depth(1014)));
channel float ch2 __attribute__((depth(1936)));
channel float ch3 __attribute__((depth(400)));
channel float ch4 __attribute__((depth(400)));
channel float ch5 __attribute__((depth(120)));
channel float ch6 __attribute__((depth(84)));
channel float ch7 __attribute__((depth(10)));

__kernel void fuse_conv2d_relu_kernel0(
    __global const float* restrict input0, 
    __global const float* restrict input1, 
    __global const float* restrict input2) 
{
  float lcompute[676];
  //float out[4056];

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 26; ++yy) {
      for (int xx = 0; xx < 26; ++xx) {
        //lcompute[((yy * 26) + xx)] = 0.000000e+00f;
        float tmp = 0.0e+00f;
        #pragma unroll
        for (int ry = 0; ry < 3; ++ry) {
          #pragma unroll
          for (int rx = 0; rx < 3; ++rx) {
            tmp += input0[((((yy + ry) * 28) + xx) + rx)] * input1[((((ax1 * 3) + ry) * 3) + rx)];
          }
        }
        lcompute[((yy * 26) + xx)] = tmp;
      }
    }
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 26; ++ax3) {
        //out[((((ax1 * 26) + ax2) * 26) + ax3)] = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
        float out = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
        write_channel_intel(ch0, out);
      }
    }
  }

  /*for (int i = 0; i < 4056; ++i) {
    write_channel_intel(ch0, out[i]);
  }*/
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fuse_avg_pool2d_kernel0() {
  //float ltensor[1014];
  float in[4056];

  for (int i = 0; i < 4056; ++i) {
      in[i] = read_channel_intel(ch0);      
  }

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 13; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 13; ++ax3) {
        //ltensor[((((ax1 * 13) + ax2) * 13) + ax3)] = 0.000000e+00f;
        float tmp = 0.0e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tmp += in[((((((((ax1 * 13) + ax2) * 2) + rv) * 13) + ax3) * 2) + rv1)] * 2.500000e-01f;
          }
        }
        //ltensor[((((ax1 * 13) + ax2) * 13) + ax3)] = tmp;
        write_channel_intel(ch1, tmp);
      }
    }
  }

  /*
  for (int i = 0; i < 1014; ++i) {
    write_channel_intel(ch1, ltensor[i]);
  }
  */
}

__kernel void fuse_conv2d_relu_1_kernel0(
    __global const float* restrict input1, 
    __global const float* restrict input2
) {
  float lcompute[121];
  float in[1014];
  //float out[1936];
  
  for (int i = 0; i < 1014; ++i) {
      in[i] = read_channel_intel(ch1);      
  }
  
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int yy = 0; yy < 11; ++yy) {
      for (int xx = 0; xx < 11; ++xx) {
        float tmp = 0.00e+00f;
        //lcompute[((yy * 11) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 6; ++rc) {
          #pragma unroll
          for (int ry = 0; ry < 3; ++ry) {
            #pragma unroll
            for (int rx = 0; rx < 3; ++rx) {
              tmp += in[((((((rc * 13) + yy) + ry) * 13) + xx) + rx)] * input1[((((((ax1 * 6) + rc) * 3) + ry) * 3) + rx)];
            }
          }
        }
        lcompute[((yy * 11) + xx)] = tmp;
      }
    }
    for (int ax2 = 0; ax2 < 11; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 11; ++ax3) {
        //out[((((ax1 * 11) + ax2) * 11) + ax3)] = max((lcompute[((ax2 * 11) + ax3)] + input2[ax1]), 0.000000e+00f);
        float out = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
        write_channel_intel(ch2, out);
      }
    }
  }
 
  /*
  for (int i = 0; i < 1924; ++i) {
    write_channel_intel(ch2, out[i]);
  }
  */
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fuse_avg_pool2d_1_kernel0() {
  //float ltensor[400];
  float in[1936];

  for (int i = 0; i < 1924; ++i) {
      in[i] = read_channel_intel(ch2);      
  }

  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        float tmp = 0.0e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tmp += in[(((((ax1 * 121) + (ax2 * 22)) + (rv * 11)) + (ax3 * 2)) + rv1)] * 2.500000e-01f;
          }
        }
        //ltensor[((((ax1 * 5) + ax2) * 5) + ax3)] = tmp;
        write_channel_intel(ch3, tmp);
      }
    }
  }
  /*
  for (int i = 0; i < 400; ++i) {
    write_channel_intel(ch3, ltensor[i]);
  }
  */
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

__kernel void fuse_dense_relu_kernel0(
    __global const float* restrict input1,
    __global const float* restrict input2
) {
  float in[400];
  //float out[120];
  for (int i = 0; i < 400; i++) {
      in[i] = read_channel_intel(ch4);      
  }

  #pragma unroll 2
  for (int ax1 = 0; ax1 < 120; ++ax1) {
    float sum = 0.000000e+00f;
    #pragma unroll 10
    for (int k = 0; k < 400; ++k) {
      sum += in[k] * input1[((ax1 * 400) + k)];
    }
    float out = max((sum + input2[ax1]), 0.000000e+00f);
    write_channel_intel(ch5, out);
  }
  
  /*
  for (int i = 0; i < 120; i++) {
    write_channel_intel(ch5, out[i]);
  }
  */
}

__kernel void fuse_dense_relu_1_kernel0(
        __global const float* restrict input1, 
        __global const float* restrict input2
) {
  float in[120];
  //float out[84];
  for (int i = 0; i < 120; i++) {
      in[i] = read_channel_intel(ch5);      
  }

  for (int ax1 = 0; ax1 < 84; ++ax1) {
    float sum = 0.000000e+00f;
    #pragma unroll 10
    for (int k = 0; k < 120; ++k) {
      sum += in[k] * input1[((ax1 * 120) + k)];
    }
    float out = max((sum + input2[ax1]), 0.000000e+00f);
    write_channel_intel(ch6, out);

  }
  
  /*
  for (int i = 0; i < 84; i++) {
    write_channel_intel(ch6, out[i]);
  }
  */
}

__kernel void fuse_dense_kernel0(
        __global const float* restrict input1, 
        __global const float* restrict input2
) {
  float in[84];
  //float out[10];
  
  for (int i = 0; i < 84; i++) {
      in[i] = read_channel_intel(ch6);      
  }

  for (int j = 0; j < 10; ++j) {
    float sum = 0.000000e+00f;
    #pragma unroll 4
    for (int k = 0; k < 84; ++k) {
      sum += in[k] * input1[((j * 84) + k)];
    }
    float out= (sum + input2[j]);
    write_channel_intel(ch7, out);
  }

  /*
  for (int i = 0; i < 10; i++) {
    write_channel_intel(ch7, out[i]);
  }
  */
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

/*
__kernel void fuse_softmax_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) { 
  float in[10];
  for (int i = 0; i < 10; i++) {
      in[i] = read_channel_intel(ch7);      
  }

  float m = -3.402823e+38f;
  #pragma unroll
  for (int i = 0; i < 10; ++i) {
    m = max(m, in[i]);
  }

  float sum = 0;
  #pragma unroll
  for (int i = 0; i < 10; ++i) {
    sum += exp(in[i] - m);
  }

  for (int i = 0; i < 10; ++i) {
    tensor2[i] = exp((in[i] - m)) / sum;
  }
}
*/

