#pragma OPENCL EXTENSION cl_intel_channels : enable

__kernel void fuse_conv2d_relu_kernel0(
    __global const float* restrict input0, 
    __global const float* restrict input1, 
    __global const float* restrict input2,
    __global float* restrict out) 
{
  local float lcompute[676];

  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 26; ++yy) {
      for (int xx = 0; xx < 26; ++xx) {
        lcompute[((yy * 26) + xx)] = 0.000000e+00f;
        #pragma unroll
        for (int ry = 0; ry < 3; ++ry) {
          #pragma unroll
          for (int rx = 0; rx < 3; ++rx) {
            lcompute[((yy * 26) + xx)] = (lcompute[((yy * 26) + xx)] + (input0[((((yy + ry) * 28) + xx) + rx)] * input1[((((ax1 * 3) + ry) * 3) + rx)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 26; ++ax3) {
        out[((((ax1 * 26) + ax2) * 26) + ax3)] = max((lcompute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_avg_pool2d_kernel0(
    __global const float* restrict in, 
    __global float* restrict out
) {
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 13; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 13; ++ax3) {
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            out[((((ax1 * 13) + ax2) * 13) + ax3)] += (in[((((((((ax1 * 13) + ax2) * 2) + rv) * 13) + ax3) * 2) + rv1)] * 2.500000e-01f);
          }
        }
      }
    }
  }
}

__kernel void fuse_conv2d_relu_1_kernel0(
    __global const float* restrict in,
    __global const float* restrict input1, 
    __global const float* restrict input2,
    __global float* restrict out
) {
  local float lcompute[121];

  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int yy = 0; yy < 11; ++yy) {
      for (int xx = 0; xx < 11; ++xx) {
        lcompute[((yy * 11) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 6; ++rc) {
          #pragma unroll
          for (int ry = 0; ry < 3; ++ry) {
            #pragma unroll
            for (int rx = 0; rx < 3; ++rx) {
              lcompute[((yy * 11) + xx)] += (in[((((((rc * 13) + yy) + ry) * 13) + xx) + rx)] * input1[((((((ax1 * 6) + rc) * 3) + ry) * 3) + rx)]);
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
}

__kernel void fuse_avg_pool2d_1_kernel0(
    __global const float* restrict in, 
    __global float* restrict out
) {
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            out[((((ax1 * 5) + ax2) * 5) + ax3)] += (in[(((((ax1 * 121) + (ax2 * 22)) + (rv * 11)) + (ax3 * 2)) + rv1)] * 2.500000e-01f);
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_flatten_kernel0(
    __global const float* restrict in, 
    __global float* restrict out
) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 400; ++ax0_ax1_fused_inner) {
    out[ax0_ax1_fused_inner] = in[(((ax0_ax1_fused_inner % 16) * 25) + (ax0_ax1_fused_inner / 16))];
  }
}

__kernel void fuse_dense_relu_kernel0(
    __global const float* restrict in, 
    __global const float* restrict input1,
    __global const float* restrict input2,
    __global float* restrict out
) {

  #pragma unroll 2
  for (int ax1 = 0; ax1 < 120; ++ax1) {
    float sum = 0.0e+00f;
    #pragma unroll 10
    for (int k = 0; k < 400; ++k) {
      sum += in[k] * input1[((ax1 * 400) + k)];
    }
    out[ax1] = max((sum + input2[ax1]), 0.000000e+00f);
  }
}

__kernel void fuse_dense_relu_1_kernel0(
    __global const float* restrict in, 
    __global const float* restrict input1, 
    __global const float* restrict input2,
    __global float* restrict out
) {

  for (int ax1 = 0; ax1 < 84; ++ax1) {
    float sum = 0.000000e+00f;
    #pragma unroll 10
    for (int k = 0; k < 120; ++k) {
      sum += in[k] * input1[((ax1 * 120) + k)];
    }
    out[ax1] = max((sum + input2[ax1]), 0.000000e+00f);
  }
}

__kernel void fuse_dense_kernel0(
    __global const float* restrict in, 
    __global const float* restrict input1, 
    __global const float* restrict input2,
    __global float* restrict out
) {
  for (int j = 0; j < 10; ++j) {
    float sum = 0.000000e+00f;
    #pragma unroll 4
    for (int k = 0; k < 84; ++k) {
      sum += in[k] * input1[((j * 84) + k)];
    }
    out[j] = (sum + input2[j]);
  }
}

__kernel void fuse_softmax_kernel0(
    __global const float* restrict in, 
    __global float* restrict tensor2
) {
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

