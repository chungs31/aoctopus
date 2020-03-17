/*
__kernel void fuse_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 151875; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f);
  }
}

// experiment 1 
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

// Input 0 marked volatile -- caches not inferred
__kernel void fuse_conv2d_kernel0_O1(__global float* restrict compute, __global volatile float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
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

// RX unrolled
__kernel void fuse_conv2d_kernel0_O2(__global float* restrict compute, __global volatile float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
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

// RY unrolled
__kernel void fuse_conv2d_kernel0_O3(__global float* restrict compute, __global volatile float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((yy * 112) + xx)] = 0.000000e+00f;
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

// Computation fully unrolled
__kernel void fuse_conv2d_kernel0_O4(__global float* restrict compute, __global volatile float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
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

*/

/* experiment 2 */
__kernel void fuse_conv2d_18_kernel0_baseline(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
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
}

__kernel void fuse_conv2d_18_kernel0_output_split(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          compute[((yy * 7) + xx)] = (compute[((yy * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)]));
        }
      }
    }
  }
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_18_kernel0_unrolled(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      #pragma unroll 
      for (int yy = 0; yy < 7; ++yy) {
        #pragma unroll 
        for (int xx = 0; xx < 7; ++xx) {
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
}

__kernel void fuse_conv2d_18_kernel0_yy_ur(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      #pragma unroll 
      for (int yy = 0; yy < 7; ++yy) {
        for (int xx = 0; xx < 7; ++xx) {
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
}

__kernel void fuse_conv2d_18_kernel0_xx_ur(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      for (int yy = 0; yy < 7; ++yy) {
      #pragma unroll 
        for (int xx = 0; xx < 7; ++xx) {
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
}

__kernel void fuse_conv2d_18_kernel0_coal0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      #pragma unroll 4
      #pragma loop_coalesce 2
      for (int yy = 0; yy < 7; ++yy) {
        for (int xx = 0; xx < 7; ++xx) {
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
}

__kernel void fuse_conv2d_18_kernel0_coal1(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
        int yy = 0;
        int xx = 0;
        //#pragma unroll 4
        while (yy < 7) {
            compute[((yy * 7) + xx)] = (compute[((yy * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)]));
            xx++;
            if (xx == 7) {
                xx = 0;
                yy++;
            }
        }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_18_kernel0_coal2(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
        int yy = 0;
        int xx = 0;
        for (int i = 0; i < 49; i++) {
            compute[((yy * 7) + xx)] = (compute[((yy * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)]));
            xx++;
            if ((i % 7) == 0 && i != 0) {
                xx = 0;
                yy++;
            }
        }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_18_kernel0_sep_ops(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((yy * 7) + xx)] = 0.000000e+00f;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      for (int yy = 0; yy < 7; ++yy) {
        for (int xx = 0; xx < 7; ++xx) {
          float sum = compute[((yy * 7) + xx)];
          float dot_p = input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)];
          compute[((yy * 7) + xx)] = sum + dot_p;
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        T_clip[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((compute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}


