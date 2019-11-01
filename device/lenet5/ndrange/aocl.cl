__kernel void fuse_conv2d_relu_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 26; ++yy) {
      for (int xx = 0; xx < 26; ++xx) {
        compute[((yy * 26) + xx)] = 0.000000e+00f;
        #pragma unroll
        for (int ry = 0; ry < 3; ++ry) {
          #pragma unroll
          for (int rx = 0; rx < 3; ++rx) {
            compute[((yy * 26) + xx)] = (compute[((yy * 26) + xx)] + (input0[((((yy + ry) * 28) + xx) + rx)] * input1[((((ax1 * 3) + ry) * 3) + rx)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      for (int ax3 = 0; ax3 < 26; ++ax3) {
        T_relu[((((ax1 * 26) + ax2) * 26) + ax3)] = max((compute[((ax2 * 26) + ax3)] + input2[ax1]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 13; ++ax2) {
      for (int ax3 = 0; ax3 < 13; ++ax3) {
        tensor[((((ax1 * 13) + ax2) * 13) + ax3)] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 13) + ax2) * 13) + ax3)] = (tensor[((((ax1 * 13) + ax2) * 13) + ax3)] + (input0[((((((((ax1 * 13) + ax2) * 2) + rv) * 13) + ax3) * 2) + rv1)] * 2.500000e-01f));
          }
        }
      }
    }
  }
}

__kernel void fuse_conv2d_relu_1_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int yy = 0; yy < 11; ++yy) {
      for (int xx = 0; xx < 11; ++xx) {
        compute[((yy * 11) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 6; ++rc) {
          #pragma unroll
          for (int ry = 0; ry < 3; ++ry) {
            #pragma unroll
            for (int rx = 0; rx < 3; ++rx) {
              compute[((yy * 11) + xx)] = (compute[((yy * 11) + xx)] + (input0[((((((rc * 13) + yy) + ry) * 13) + xx) + rx)] * input1[((((((ax1 * 6) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 11; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 11; ++ax3) {
        T_relu[((((ax1 * 11) + ax2) * 11) + ax3)] = max((compute[((ax2 * 11) + ax3)] + input2[ax1]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_avg_pool2d_1_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        tensor[((((ax1 * 5) + ax2) * 5) + ax3)] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          #pragma unroll
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 5) + ax2) * 5) + ax3)] = (tensor[((((ax1 * 5) + ax2) * 5) + ax3)] + (input0[(((((ax1 * 121) + (ax2 * 22)) + (rv * 11)) + (ax3 * 2)) + rv1)] * 2.500000e-01f));
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_flatten_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 400; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = input0[(((ax0_ax1_fused_inner % 16) * 25) + (ax0_ax1_fused_inner / 16))];
  }
}

__kernel void fuse_dense_relu_kernel0(__global float* restrict T_dense, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2) {
  int ax1 = get_global_id(0);

  float sum = 0.0e+00f;
  // 120 outer loop
  for (int k = 0; k < 400; ++k) {
    sum += input0[k] * input1[(ax1*400) + k];
  }

  T_relu[ax1] = max(sum + input2[ax1], 0.0e+00f);
}

__kernel void fuse_dense_relu_1_kernel0(__global float* restrict T_dense, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2) {
  /*
  for (int ax1 = 0; ax1 < 84; ++ax1) {
    T_dense[0] = 0.000000e+00f;
    for (int k = 0; k < 120; ++k) {
      T_dense[0] = (T_dense[0] + (input0[k] * input1[((ax1 * 120) + k)]));
    }
    T_relu[ax1] = max((T_dense[0] + input2[ax1]), 0.000000e+00f);
  }
  */
  
  int ax1 = get_global_id(0);

  float sum = 0.0e+00f;
  // 84 outer loop
  for (int k = 0; k < 120; ++k) {
    sum += input0[k] * input1[(ax1*120) + k];
  }

  T_relu[ax1] = max(sum + input2[ax1], 0.0e+00f);
}

__kernel void fuse_dense_kernel0(__global float* restrict T_dense, __global float* restrict input0, __global float* restrict input1, __global float* restrict compute, __global float* restrict input2) {
  /*for (int j = 0; j < 10; ++j) {
    T_dense[0] = 0.000000e+00f;
    for (int k = 0; k < 84; ++k) {
      T_dense[0] = (T_dense[0] + (input0[k] * input1[((j * 84) + k)]));
    }
    compute[j] = (T_dense[0] + input2[j]);
  }*/
  int j = get_global_id(0);

  float sum = 0.0e+00f;
  // 120 outer loop
  for (int k = 0; k < 84; ++k) {
    sum += input0[k] * input1[(j*84) + k];
  }

  compute[j] = sum + input2[j];
}

__kernel void fuse_softmax_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
  for (int ax1 = 0; ax1 < 10; ++ax1) {
    tensor[0] = -3.402823e+38f;
    #pragma unroll
    for (int k1 = 0; k1 < 10; ++k1) {
      tensor[0] = max(tensor[0], input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    #pragma unroll
    for (int k2 = 0; k2 < 10; ++k2) {
      tensor1[0] = (tensor1[0] + exp((input0[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor[0])) / tensor1[0]);
  }
}

