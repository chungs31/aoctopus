
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float ch1 __attribute__((depth(512)));
channel float ch2 __attribute__((depth(512)));
channel float ch3 __attribute__((depth(512)));
channel float ch4 __attribute__((depth(512)));
channel float ch5 __attribute__((depth(512)));
channel float ch6 __attribute__((depth(512)));
channel float ch7 __attribute__((depth(512)));
channel float ch8 __attribute__((depth(512)));
channel float ch9 __attribute__((depth(512)));
channel float ch10 __attribute__((depth(512)));
channel float ch11 __attribute__((depth(512)));
channel float ch12 __attribute__((depth(512)));
channel float ch13 __attribute__((depth(512)));
channel float ch14 __attribute__((depth(512)));
channel float ch15 __attribute__((depth(512)));
channel float ch16 __attribute__((depth(512)));
channel float ch17 __attribute__((depth(512)));
channel float ch18 __attribute__((depth(512)));
channel float ch19 __attribute__((depth(512)));
channel float ch20 __attribute__((depth(512)));
channel float ch21 __attribute__((depth(512)));
channel float ch22 __attribute__((depth(512)));
channel float ch23 __attribute__((depth(512)));
channel float ch24 __attribute__((depth(512)));
channel float ch25 __attribute__((depth(512)));
channel float ch26 __attribute__((depth(512)));
channel float ch27 __attribute__((depth(512)));
channel float ch28 __attribute__((depth(512)));
channel float ch29 __attribute__((depth(512)));
channel float ch30 __attribute__((depth(512)));
channel float ch31 __attribute__((depth(512)));
channel float ch32 __attribute__((depth(512)));
channel float ch33 __attribute__((depth(512)));
channel float ch34 __attribute__((depth(512)));
channel float ch35 __attribute__((depth(512)));
channel float ch36 __attribute__((depth(512)));

__kernel void fuse_pad_kernel0(__global float* restrict in) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 151875; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    //T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f);
    write_channel_intel(ch1, (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f) );
  }
}

__kernel void fuse_conv2d_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[151875];
  float lcompute[12544];
  for (int i = 0; i < 151875; ++i) {
      in[i] = read_channel_intel(ch1);
  }

  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        //lcompute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              lcompute[((yy * 112) + xx)] = (lcompute[((yy * 112) + xx)] + (in[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        //out[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch2, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 401408; ++i) {
  //  write_channel_intel(ch2, out[i]);
  //}
}

__kernel void fuse_conv2d_1_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[401408];
  float lcompute[12544];
  for (int i = 0; i < 401408; ++i) {
      in[i] = read_channel_intel(ch2);
  }

  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int i = 0; i < 112; ++i) {
      for (int j = 0; j < 112; ++j) {
        //lcompute[((i * 112) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 112) + j)] = (lcompute[((i * 112) + j)] + ((float)((((((1 - di) <= i) && (i < (113 - di))) && ((1 - dj) <= j)) && (j < (113 - dj))) ? in[(((((((ax1 * 112) + i) + di) * 112) + j) + dj) + -113)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        //out[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch3, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 401408; ++i) {
  //  write_channel_intel(ch3, out[i]);
  //}
}

__kernel void fuse_conv2d_2_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[401408];
  float lcompute[12544];
  for (int i = 0; i < 401408; ++i) {
      in[i] = read_channel_intel(ch3);
  }

  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        //lcompute[((yy * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          lcompute[((yy * 112) + xx)] = (lcompute[((yy * 112) + xx)] + (in[((((rc * 112) + yy) * 112) + xx)] * input1[((ax1 * 32) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 112; ++ax3) {
        //out[((((ax1 * 112) + ax2) * 112) + ax3)] = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 112) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch4, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 802816; ++i) {
  //  write_channel_intel(ch4, out[i]);
  //}
}

__kernel void fuse_pad_1_kernel0() {
  float in[802816];
  for (int i = 0; i < 802816; ++i) {
      in[i] = read_channel_intel(ch4);
  }
  
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 817216; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    //out[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) < 12656) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113) < 112)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12769) * 112) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) / 113)) * 112) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113))] : 0.000000e+00f);
    write_channel_intel(ch5, (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) < 12656) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113) < 112)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12769) * 112) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) / 113)) * 112) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113))] : 0.000000e+00f));
  }
}

__kernel void fuse_conv2d_3_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[817216];
  float lcompute[3136];
  //float out[200704];
  for (int i = 0; i < 817216; ++i) {
      in[i] = read_channel_intel(ch5);
  }

  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int i = 0; i < 56; ++i) {
      for (int j = 0; j < 56; ++j) {
        //lcompute[((i * 56) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 56) + j)] = (lcompute[((i * 56) + j)] + (in[(((((ax1 * 12769) + (i * 226)) + (di * 113)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        //out[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch6, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 200704; ++i) {
  //  write_channel_intel(ch6, out[i]);
  // }
}

__kernel void fuse_conv2d_4_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[200704];
  float lcompute[3136];
  //float out[401408];
  for (int i = 0; i < 200704; ++i) {
      in[i] = read_channel_intel(ch6);
  }

  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        //lcompute[((yy * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          lcompute[((yy * 56) + xx)] = (lcompute[((yy * 56) + xx)] + (in[((((rc * 56) + yy) * 56) + xx)] * input1[((ax1 * 64) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        //out[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch7, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 401408; ++i) {
  //  write_channel_intel(ch7, out[i]);
 // }
}

__kernel void fuse_conv2d_5_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[401408];
  float lcompute[3136];
  //float out[401408];
  for (int i = 0; i < 401408; ++i) {
      in[i] = read_channel_intel(ch7);
  }

  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int i = 0; i < 56; ++i) {
      for (int j = 0; j < 56; ++j) {
        //lcompute[((i * 56) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 56) + j)] = (lcompute[((i * 56) + j)] + ((float)((((((1 - di) <= i) && (i < (57 - di))) && ((1 - dj) <= j)) && (j < (57 - dj))) ? in[(((((((ax1 * 56) + i) + di) * 56) + j) + dj) + -57)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        //out[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch8, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 401408; ++i) {
   // write_channel_intel(ch8, out[i]);
  //}
}

__kernel void fuse_conv2d_6_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[401408];
  float lcompute[3136];
  //float out[401408];
  for (int i = 0; i < 401408; ++i) {
      in[i] = read_channel_intel(ch8);
  }
  
  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        //lcompute[((yy * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          lcompute[((yy * 56) + xx)] = (lcompute[((yy * 56) + xx)] + (in[((((rc * 56) + yy) * 56) + xx)] * input1[((ax1 * 128) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        //out[((((ax1 * 56) + ax2) * 56) + ax3)] = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 56) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch9, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 401408; ++i) {
  //  write_channel_intel(ch9, out[i]);
  //}
}

__kernel void fuse_pad_2_kernel0() {
  float in[401408];
  for (int i = 0; i < 401408; ++i) {
      in[i] = read_channel_intel(ch9);
  }
  
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 415872; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    write_channel_intel(ch10, (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) < 3192) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57) < 56)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3249) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) / 57)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57))] : 0.000000e+00f));
    //out[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) < 3192) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57) < 56)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3249) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) / 57)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57))] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_7_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[415872];
  float lcompute[784];
  //float out[100352];
  for (int i = 0; i < 415872; ++i) {
      in[i] = read_channel_intel(ch10);
  }

  for (int ax1 = 0; ax1 < 128; ++ax1) {
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        //lcompute[((i * 28) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 28) + j)] = (lcompute[((i * 28) + j)] + (in[(((((ax1 * 3249) + (i * 114)) + (di * 57)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        //out[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch11, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch11, out[i]);
  //}
}

__kernel void fuse_conv2d_8_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[784];
  //float out[200704];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch11);
  }

  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        //lcompute[((yy * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          lcompute[((yy * 28) + xx)] = (lcompute[((yy * 28) + xx)] + (in[((((rc * 28) + yy) * 28) + xx)] * input1[((ax1 * 128) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        //out[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch12, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 200704; ++i) {
  // write_channel_intel(ch12, out[i]);
  //}
}

__kernel void fuse_conv2d_9_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[200704];
  float lcompute[784];
  //float out[200704];
  for (int i = 0; i < 200704; ++i) {
      in[i] = read_channel_intel(ch12);
  }

  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        //lcompute[((i * 28) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 28) + j)] = (lcompute[((i * 28) + j)] + ((float)((((((1 - di) <= i) && (i < (29 - di))) && ((1 - dj) <= j)) && (j < (29 - dj))) ? in[(((((((ax1 * 28) + i) + di) * 28) + j) + dj) + -29)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        //out[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch13, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 200704; ++i) {
  //  write_channel_intel(ch13, out[i]);
  //}
}

__kernel void fuse_conv2d_10_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[200704];
  float lcompute[784];
  //float out[200704];
  for (int i = 0; i < 200704; ++i) {
      in[i] = read_channel_intel(ch13);
  }

  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        //lcompute[((yy * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          lcompute[((yy * 28) + xx)] = (lcompute[((yy * 28) + xx)] + (in[((((rc * 28) + yy) * 28) + xx)] * input1[((ax1 * 256) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        //out[((((ax1 * 28) + ax2) * 28) + ax3)] = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 28) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch14, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 200704; ++i) {
  //  write_channel_intel(ch14, out[i]);
  //}
}

__kernel void fuse_pad_3_kernel0() {
  float in[200704];
  for (int i = 0; i < 200704; ++i) {
      in[i] = read_channel_intel(ch14);
  
  }

  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    write_channel_intel(ch15, (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) < 812) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29) < 28)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 841) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) / 29)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29))] : 0.000000e+00f));
  }
}

__kernel void fuse_conv2d_11_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[215296];
  float lcompute[196];
  //float out[50176];
  for (int i = 0; i < 215296; ++i) {
      in[i] = read_channel_intel(ch15);
  }

  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + (in[(((((ax1 * 841) + (i * 58)) + (di * 29)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch16, tmp);

      }
    }
  }
  
  //for (int i = 0; i < 50176; ++i) {
  //  write_channel_intel(ch16, out[i]);
  //}
}

__kernel void fuse_conv2d_12_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[50176];
  float lcompute[196];
  //float out[100352];
  for (int i = 0; i < 50176; ++i) {
      in[i] = read_channel_intel(ch16);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 256) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch17, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch17, out[i]);
  //}
}

__kernel void fuse_conv2d_13_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
  //float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch17);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? in[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch18, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
   // write_channel_intel(ch18, out[i]);
  //}
}

__kernel void fuse_conv2d_14_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
  //float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch18);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch19, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch19, out[i]);
  //}
}

__kernel void fuse_conv2d_13_kernel1(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
  //float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch19);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? in[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch20, tmp);
      }
    }
  }
  
//  for (int i = 0; i < 100352; ++i) {
//    write_channel_intel(ch20, out[i]);
//  }
}

__kernel void fuse_conv2d_14_kernel1(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
  //float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch20);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch21, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch21, out[i]);
  //}
}

__kernel void fuse_conv2d_13_kernel2(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch21);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? in[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch22, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch22, out[i]);
  //}
}

__kernel void fuse_conv2d_14_kernel2(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch22);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch23, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch23, out[i]);
  //}
}

__kernel void fuse_conv2d_13_kernel3(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch23);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? in[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch24, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch24, out[i]);
  //}
}

__kernel void fuse_conv2d_14_kernel3(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch24);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch25, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch25, out[i]);
  //}
}

__kernel void fuse_conv2d_13_kernel4(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch25);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 14; ++i) {
      for (int j = 0; j < 14; ++j) {
        //lcompute[((i * 14) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 14) + j)] = (lcompute[((i * 14) + j)] + ((float)((((((1 - di) <= i) && (i < (15 - di))) && ((1 - dj) <= j)) && (j < (15 - dj))) ? in[(((((((ax1 * 14) + i) + di) * 14) + j) + dj) + -15)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch26, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch26, out[i]);
  //}
}

__kernel void fuse_conv2d_14_kernel4(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[100352];
  float lcompute[196];
//  float out[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch26);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        //lcompute[((yy * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 14) + xx)] = (lcompute[((yy * 14) + xx)] + (in[((((rc * 14) + yy) * 14) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        //out[((((ax1 * 14) + ax2) * 14) + ax3)] = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 14) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch27, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 100352; ++i) {
  //  write_channel_intel(ch27, out[i]);
  //}
}


__kernel void fuse_pad_4_kernel0() {
  float in[100352];
  for (int i = 0; i < 100352; ++i) {
      in[i] = read_channel_intel(ch27);
  }
  
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    write_channel_intel(ch28, (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 210) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15) < 14)) ? in[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 225) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) / 15)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15))] : 0.000000e+00f));
  }
}

__kernel void fuse_conv2d_15_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[115200];
  float lcompute[196];
  //float out[25088];
  for (int i = 0; i < 115200; ++i) {
      in[i] = read_channel_intel(ch28);
  }

  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        //lcompute[((i * 7) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 7) + j)] = (lcompute[((i * 7) + j)] + (in[(((((ax1 * 225) + (i * 30)) + (di * 15)) + (j * 2)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        //out[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch29, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 25088; ++i) {
  //  write_channel_intel(ch29, out[i]);
  //}
}

__kernel void fuse_conv2d_16_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[25088];
  float lcompute[49];
  //float out[50176];
  for (int i = 0; i < 25088; ++i) {
      in[i] = read_channel_intel(ch29);
  }

  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        //lcompute[((yy * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          lcompute[((yy * 7) + xx)] = (lcompute[((yy * 7) + xx)] + (in[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 512) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        //out[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch30, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 50176; ++i) {
  //  write_channel_intel(ch30, out[i]);
  //}
}

__kernel void fuse_conv2d_17_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[50176];
  float lcompute[49];
  //float out[50176];
  for (int i = 0; i < 50176; ++i) {
      in[i] = read_channel_intel(ch30);
  }

  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        //lcompute[((i * 7) + j)] = 0.000000e+00f;
        for (int di = 0; di < 3; ++di) {
          for (int dj = 0; dj < 3; ++dj) {
            lcompute[((i * 7) + j)] = (lcompute[((i * 7) + j)] + ((float)((((((1 - di) <= i) && (i < (8 - di))) && ((1 - dj) <= j)) && (j < (8 - dj))) ? in[(((((((ax1 * 7) + i) + di) * 7) + j) + dj) + -8)] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        //out[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch31, tmp);
      }
    }
  }
  
  //for (int i = 0; i < 50176; ++i) {
  //  write_channel_intel(ch31, out[i]);
  //}
}

__kernel void fuse_conv2d_18_kernel0(__global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  float in[50176];
  float lcompute[49];
  //float out[50176];
  for (int i = 0; i < 50176; ++i) {
      in[i] = read_channel_intel(ch31);
  }

  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        //lcompute[((yy * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          lcompute[((yy * 7) + xx)] = (lcompute[((yy * 7) + xx)] + (in[((((rc * 7) + yy) * 7) + xx)] * input1[((ax1 * 1024) + rc)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        //out[((((ax1 * 7) + ax2) * 7) + ax3)] = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        float tmp = max(min(((lcompute[((ax2 * 7) + ax3)] * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
        write_channel_intel(ch32, tmp);

      }
    }
  }
  
  //for (int i = 0; i < 50176; ++i) {
  //  write_channel_intel(ch32, out[i]);
  //}
}

__kernel void fuse_global_avg_pool2d_kernel0() {
  float in[50176];
  float tensor[1024];
  for (int i = 0; i < 50176; ++i) {
      in[i] = read_channel_intel(ch32);
  }
  
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    //tensor[ax1] = 0.000000e+00f;
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      for (int rv2 = 0; rv2 < 7; ++rv2) {
        tensor[ax1] = (tensor[ax1] + (in[((((ax1 * 7) + rv1) * 7) + rv2)] * 2.040816e-02f));
      }
    }
  }
  
  for (int i = 0; i < 1024; ++i) {
    write_channel_intel(ch33, tensor[i]);
  }
}

__kernel void fuse_transpose_flatten_reshape_kernel0() {
  /*float in[1024];
  for (int i = 0; i < 1024; ++i) {
      in[i] = read_channel_intel(ch33);
  }*/

  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1024; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    float tmp = read_channel_intel(ch33);
    write_channel_intel(ch34, tmp);
  }
  
}

__kernel void fuse_conv2d_kernel1(__global float* restrict input1, __global float* restrict input2) {
  float in[1024];
  float out[1000];
  for (int i = 0; i < 1024; ++i) {
      in[i] = read_channel_intel(ch34);
  }

  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    float tmp = 0.000000e+00f;
    for (int rc = 0; rc < 1024; ++rc) {
      tmp = (tmp + (in[rc] * input1[((ax1 * 1024) + rc)]));
    }
    out[ax1] = (tmp + input2[ax1]);
  }
  
  for (int i = 0; i < 1000; ++i) {
    write_channel_intel(ch35, out[i]);
  }
}

__kernel void fuse_reshape_kernel0() {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1000; ++ax0_ax1_fused_inner) {
    //T_reshape[ax0_ax1_fused_inner] = in[ax0_ax1_fused_inner];
    float in = read_channel_intel(ch35);
    write_channel_intel(ch36, in);
  }
}

__kernel void fuse_softmax_kernel0(__global float* restrict tensor2) {
  float in[1000];
  float out[1000];
  for (int i = 0; i < 1000; ++i) {
      in[i] = read_channel_intel(ch36);
  }

  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    float tmp0 = -3.402823e+38f;
    for (int k1 = 0; k1 < 1000; ++k1) {
      tmp0 = max(tmp0, in[k1]);
    }
    float tmp1 = 0.000000e+00f;
    for (int k2 = 0; k2 < 1000; ++k2) {
      tmp1 = (tmp1 + exp((in[k2] - tmp0)));
    }
    tensor2[ax1] = (exp((in[ax1] - tmp0)) / tmp1);
  }
}

