
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int ch1  __attribute__((depth(1)));
channel int ch2  __attribute__((depth(1)));
channel int ch3  __attribute__((depth(1)));
channel int ch4  __attribute__((depth(1)));
channel int ch5  __attribute__((depth(1)));
channel int ch6  __attribute__((depth(1)));
channel int ch7  __attribute__((depth(1)));
channel int ch8  __attribute__((depth(1)));
channel int ch9  __attribute__((depth(1)));
channel int ch10 __attribute__((depth(1)));
channel int ch11 __attribute__((depth(1)));
channel int ch12 __attribute__((depth(1)));
channel int ch13 __attribute__((depth(1)));
channel int ch14 __attribute__((depth(1)));
channel int ch15 __attribute__((depth(1)));
channel int ch16 __attribute__((depth(1)));
channel int ch17 __attribute__((depth(1)));
channel int ch18 __attribute__((depth(1)));
channel int ch19 __attribute__((depth(1)));
channel int ch20 __attribute__((depth(1)));
channel int ch21 __attribute__((depth(1)));
channel int ch22 __attribute__((depth(1)));
channel int ch23 __attribute__((depth(1)));
channel int ch24 __attribute__((depth(1)));
channel int ch25 __attribute__((depth(1)));
channel int ch26 __attribute__((depth(1)));
channel int ch27 __attribute__((depth(1)));
channel int ch28 __attribute__((depth(1)));
channel int ch29 __attribute__((depth(1)));
channel int ch30 __attribute__((depth(1)));
channel int ch31 __attribute__((depth(1)));
channel int ch32 __attribute__((depth(1)));
channel int ch33 __attribute__((depth(1)));
channel int ch34 __attribute__((depth(1)));
channel int ch35 __attribute__((depth(1)));
channel int ch36 __attribute__((depth(1)));

__kernel void fuse_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 151875; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch1, 1);
}

__kernel void fuse_conv2d_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch1);
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
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch2, 1);
}

__kernel void fuse_conv2d_1_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch2);
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
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch3, 1);
}

__kernel void fuse_conv2d_2_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch3);
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          compute[((yy * 112) + xx)] = (compute[((yy * 112) + xx)] + (input0[((((rc * 112) + yy) * 112) + xx)] * input1[((ax1 * 32) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        T_clip[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((compute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch4, 1);
}

__kernel void fuse_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  int dummy = read_channel_intel(ch4);
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 817216; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) < 12656) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113) < 112)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12769) * 112) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) / 113)) * 112) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113))] : 0.000000e+00f);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch5, 1);
}

__kernel void fuse_conv2d_3_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch5);
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int i = 0; i < 56; ++i) {
      for (int j = 0; j < 56; ++j) {
        DepthwiseConv2d[((i * 56) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 56) + j)] = (DepthwiseConv2d[((i * 56) + j)] + (input0[(((((ax1 * 12769) + (i * 226)) + (di * 113)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        T_clip[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch6, 1);
}

__kernel void fuse_conv2d_4_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch6);
  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((yy * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((yy * 56) + xx)] = (compute[((yy * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ax1 * 64) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        T_clip[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((compute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch7, 1);
}

__kernel void fuse_conv2d_5_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch7);
  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int i = 0; i < 56; ++i) {
      for (int j = 0; j < 56; ++j) {
        DepthwiseConv2d[((i * 56) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 56) + j)] = (DepthwiseConv2d[((i * 56) + j)] + ((float)((((((1 - di) <= i) && (i < (57 - di))) && ((1 - dj) <= j)) && (j < (57 - dj))) ? input0[(((((((ax1 * 56) + i) + di) * 56) + j) + dj) + -57)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        T_clip[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch8, 1);
}

__kernel void fuse_conv2d_6_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch8);
  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((yy * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          compute[((yy * 56) + xx)] = (compute[((yy * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ax1 * 128) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        T_clip[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((compute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch9, 1);
}

__kernel void fuse_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  int dummy = read_channel_intel(ch9);
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 415872; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) < 3192) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57) < 56)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3249) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) / 57)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57))] : 0.000000e+00f);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch10, 1);
}

__kernel void fuse_conv2d_7_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch10);
  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        DepthwiseConv2d[((i * 28) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 28) + j)] = (DepthwiseConv2d[((i * 28) + j)] + (input0[(((((ax1 * 3249) + (i * 114)) + (di * 57)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        T_clip[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch11, 1);
}

__kernel void fuse_conv2d_8_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch11);
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((yy * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          compute[((yy * 28) + xx)] = (compute[((yy * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ax1 * 128) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        T_clip[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((compute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch12, 1);
}

__kernel void fuse_conv2d_9_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch12);
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        DepthwiseConv2d[((i * 28) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 28) + j)] = (DepthwiseConv2d[((i * 28) + j)] + ((float)((((((1 - di) <= i) && (i < (29 - di))) && ((1 - dj) <= j)) && (j < (29 - dj))) ? input0[(((((((ax1 * 28) + i) + di) * 28) + j) + dj) + -29)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        T_clip[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch13, 1);
}

__kernel void fuse_conv2d_10_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch13);
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((yy * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((yy * 28) + xx)] = (compute[((yy * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ax1 * 256) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        T_clip[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((compute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch14, 1);
}

__kernel void fuse_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  int dummy = read_channel_intel(ch14);
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) < 812) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29) < 28)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 841) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) / 29)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29))] : 0.000000e+00f);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch15, 1);
}

__kernel void fuse_conv2d_11_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch15);
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + (input0[(((((ax1 * 841) + (i * 58)) + (di * 29)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch16, 1);
}

__kernel void fuse_conv2d_12_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch16);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 256) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch17, 1);
}

__kernel void fuse_conv2d_13_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch17);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? input0[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch18, 1);
}

__kernel void fuse_conv2d_14_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch18);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch19, 1);
}

__kernel void fuse_conv2d_13_kernel1(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch19);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? input0[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch20, 1);
}

__kernel void fuse_conv2d_14_kernel1(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch20);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch21, 1);
}

__kernel void fuse_conv2d_13_kernel2(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch21);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? input0[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch22, 1);
}

__kernel void fuse_conv2d_14_kernel2(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch22);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch23, 1);
}

__kernel void fuse_conv2d_13_kernel3(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch23);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? input0[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch24, 1);
}

__kernel void fuse_conv2d_14_kernel3(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch24);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch25, 1);
}

__kernel void fuse_conv2d_13_kernel4(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch25);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        DepthwiseConv2d[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 14) + j)] = (DepthwiseConv2d[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? input0[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch26, 1);
}

__kernel void fuse_conv2d_14_kernel4(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch26);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 14) + xx)] = (compute[((yy * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        T_clip[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((compute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch27, 1);
}


__kernel void fuse_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  int dummy = read_channel_intel(ch27);
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 210) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15) < 14)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 225) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) / 15)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15))] : 0.000000e+00f);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch28, 1);
}

__kernel void fuse_conv2d_15_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch28);
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        DepthwiseConv2d[((i * 7) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 7) + j)] = (DepthwiseConv2d[((i * 7) + j)] + (input0[(((((ax1 * 225) + (i * 30)) + (di * 15)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch29, 1);
}

__kernel void fuse_conv2d_16_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch29);
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((yy * 7) + xx)] = (compute[((yy * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch30, 1);
}

__kernel void fuse_conv2d_17_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch30);
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        DepthwiseConv2d[((i * 7) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            DepthwiseConv2d[((i * 7) + j)] = (DepthwiseConv2d[((i * 7) + j)] + ((float)((((((1 - di) <= i) && (i < (8 - di))) && ((1 - dj) <= j)) && (j < (8 - dj))) ? input0[(((((((ax1 * 7) + i) + di) * 7) + j) + dj) + -8)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((DepthwiseConv2d[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch31, 1);
}

__kernel void fuse_conv2d_18_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  int dummy = read_channel_intel(ch31);
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          compute[((yy * 7) + xx)] = (compute[((yy * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch32, 1);
}

__kernel void fuse_global_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  int dummy = read_channel_intel(ch32);
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    tensor[ax1] = 0.000000e+00f;
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      for (int rv2 = 0; rv2 < 7; ++rv2) {
        tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv1) * 7) + rv2)] * 2.040816e-02f));
      }
    }
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch33, 1);
}

__kernel void fuse_transpose_flatten_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict input0) {
  int dummy = read_channel_intel(ch33);
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1024; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_reshape[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch34, 1);
}

__kernel void fuse_conv2d_kernel1(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  int dummy = read_channel_intel(ch34);
  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    compute[0] = 0.000000e+00f;
    for (int rc = 0; rc < 1024; ++rc) {
      compute[0] = (compute[0] + (input0[rc] * input1[((ax1 * 1024) + rc)]));
    }
    T_add[ax1] = (compute[0] + input2[ax1]);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch35, 1);
}

__kernel void fuse_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict input0) {
  int dummy = read_channel_intel(ch35);
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1000; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
  write_channel_intel(ch36, 1);
}

__kernel void fuse_softmax_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
  int dummy = read_channel_intel(ch36);
  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    tensor[0] = -3.402823e+38f;
    for (int k1 = 0; k1 < 1000; ++k1) {
      tensor[0] = max(tensor[0], input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1000; ++k2) {
      tensor1[0] = (tensor1[0] + exp((input0[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor[0])) / tensor1[0]);
  }
}

