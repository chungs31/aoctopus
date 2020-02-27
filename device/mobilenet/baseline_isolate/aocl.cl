__kernel void fuse_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 151875; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_kernel0_baseline(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((yy * 112) + xx)] = (compute[((yy * 112) + xx)] + (input0[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((compute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_kernel0_unrolled(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
#pragma unroll 
        for (int rc = 0; rc < 3; ++rc) {
#pragma unroll 
          for (int ry = 0; ry < 3; ++ry) {
#pragma unroll 
            for (int rx = 0; rx < 3; ++rx) {
              compute[((yy * 112) + xx)] = (compute[((yy * 112) + xx)] + (input0[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((compute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_kernel0_loopcoal(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((yy * 112) + xx)] = (compute[((yy * 112) + xx)] + (input0[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
#pragma loop_coalesce 2 
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((compute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_kernel0_local_buf(__global float* restrict compute, __global const volatile float* restrict input0, __global const float* restrict input1, __global float* restrict T_clip, __global const float* restrict input2, __global const float* restrict input3) {
  float input0_buf[151875];
  for (int i = 0; i < 151874; i++) {
    input0_buf[i] = input0[i];
  }

  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((yy * 112) + xx)] += (input0_buf[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)]);
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((compute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}


__kernel void fuse_conv2d_1_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int i = 0; i < 112; ++i) {
      for (int j = 0; j < 112; ++j) {
        DepthwiseConv2d[((i * 112) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 112) + j)] = (DepthwiseConv2d[((i * 112) + j)] + ((float)((((((1 - di) <= i) && (i < (113 - di))) && ((1 - dj) <= j)) && (j < (113 - dj))) ? input0[(((((((ax1 * 112) + i) + di) * 112) + j) + dj) + -113)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}
