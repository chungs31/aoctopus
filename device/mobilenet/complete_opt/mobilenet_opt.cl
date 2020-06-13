#pragma OPENCL EXTENSION cl_intel_channels : enable

//#define Tr 4
#define Tc 7
#define Tm 32
#define Tn 4

// 3x3 conv
__kernel void fuse_conv2d_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3,
  int ax1_bound, int yy_bound, int xx_bound, int rc_bound) {
  for (int ax1 = 0; ax1 < 32; ++ax1) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        float tmp = 0.000000e+00f;
        #pragma unroll
        for (int rc = 0; rc < 3; ++rc) {
          #pragma unroll
          for (int ry = 0; ry < 3; ++ry) {
            #pragma unroll
            for (int rx = 0; rx < 3; ++rx) {
              tmp += input0[(((((rc * 50625) + (yy * 450)) + (ry * 225)) + (xx * 2)) + rx)] * input1[((((((ax1 * 3) + rc) * 3) + ry) * 3) + rx)];
            }
          }
        }
        T_clip[((((ax1 * 112) + yy) * 112) + xx)] = max(min(((tmp * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

// 3x3 depthwise conv A
__kernel void fuse_conv2d_1_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3,
  int ax1_bound, int yy_bound, int xx_bound, int rc_bound) {
  for (int ax1 = 0; ax1 < ax1_bound; ++ax1) {
    for (int i = 0; i < yy_bound; ++i) {
      for (int j = 0; j < yy_bound; ++j) {
        float tmp = 0.0f;
        #pragma unroll
        for (int di = 0; di < 3; ++di) {
          #pragma unroll
          for (int dj = 0; dj < 3; ++dj) {
            tmp += (float)((((((1 - di) <= i) && (i < ((yy_bound+1) - di))) && ((1 - dj) <= j)) && (j < ((yy_bound+1) - dj))) ? input0[(((((((ax1 * yy_bound) + i) + di) * yy_bound) + j) + dj) + -(yy_bound+1))] : 0.000000e+00f) * input1[((((ax1 * 3) + di) * 3) + dj)];
          }
        }
        T_clip[((((ax1 * yy_bound) + i) * yy_bound) + j)] = max(min(((tmp * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

// 3x3 depthwise conv B
__kernel void fuse_conv2d_3_kernel0(__global float* restrict DepthwiseConv2d, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3,
  int ax1_bound, int yy_bound, int ax1_stride, int i_stride, int j_stride, int di_stride) {
  for (int ax1 = 0; ax1 < ax1_bound; ++ax1) {
    for (int i = 0; i < yy_bound; ++i) {
      for (int j = 0; j < yy_bound; ++j) {
        float tmp = 0.000000e+00f;
        #pragma unroll
        for (int di = 0; di < 3; ++di) {
          #pragma unroll
          for (int dj = 0; dj < 3; ++dj) {
            tmp += input0[(((((ax1 * ax1_stride) + (i * i_stride)) + (di * di_stride)) + (j * j_stride)) + dj)] * input1[((((ax1 * 3) + di) * 3) + dj)];
          }
        }
        T_clip[((((ax1 * yy_bound) + i) * yy_bound) + j)] = max(min(((tmp * input2[ax1]) + input3[ax1]), 6.000000e+00f), 0.000000e+00f);
      }
    }
  }
}

// 1x1 conv
kernel void fuse_conv2d_rc(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_clip, __global float* restrict input2, __global float* restrict input3,
  int ax1_bound, int yy_bound, int xx_bound, int rc_bound) {
    float  weight_buf[Tm][Tn]; // This doesn't need to be banked as accesses are sequential. Very-wide accesses to weight_buf
    float  input_fm[Tn][Tc];
    //float output_p_sums[Tm][Tr][Tc]; // Support Tm parallel accesses

    for (int ax1 = 0; ax1 < ax1_bound; ax1 += Tm) {
      for (int yy = 0; yy < yy_bound; yy++) {
        for (int xx = 0; xx < yy_bound; xx += Tc) {
          float  p_sums[Tm][Tc];// = {0.0f};
          for (int rc = 0; rc < rc_bound; rc += Tn) {
            // load weights
            #pragma unroll
            for (int too = 0; too < Tm; too++) {
              #pragma unroll
              for (int tii = 0; tii < Tn; tii++) {
                weight_buf[too][tii] = input1[(ax1 + too) * rc_bound + (rc + tii)];
              }
            }
            // load input feature maps
            #pragma unroll
            for (int tii = 0; tii < Tn; tii++) { // rc
                #pragma unroll
                for (int tcc = 0; tcc < Tc; tcc++) { //  xx
                  input_fm[tii][tcc] = input0[(((((rc + tii) * yy_bound) + (yy)) * yy_bound) + (xx + tcc))];
                }
            }

            // perform MACs
            #pragma unroll
            for (int tcc = 0; tcc < Tc; tcc++) { //  xx
              #pragma unroll
              for (int too = 0; too < Tm; too++) { // ax1
                float acc_i = 0.0f;
                #pragma unroll
                for (int tii = 0; tii < Tn; tii++) { // rc
                  acc_i += weight_buf[too][tii] * input_fm[tii][tcc];
                }
                p_sums[too][tcc] = (rc == 0) ? acc_i : p_sums[too][tcc] + acc_i;
              }
            }
          }
          for (int too = 0; too < Tm; too++) {
              #pragma unroll
              for (int tcc = 0; tcc < Tc; tcc++) {
                T_clip[(((((too+ax1) * yy_bound) + (yy)) * yy_bound) + (xx+tcc))] = max(min(((p_sums[too][tcc]  * input2[ax1 + too]) + input3[ax1 + too]), 6.000000e+00f), 0.000000e+00f);
              }
          }
        }
      }
  }
}

// padding layers
__kernel void fuse_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 151875; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) < 50400) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 224)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50625) * 224) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 50625) / 225)) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225))] : 0.000000e+00f);
  }
}

__kernel void fuse_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 817216; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) < 12656) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113) < 112)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12769) * 112) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 12769) / 113)) * 112) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 113))] : 0.000000e+00f);
  }
}

__kernel void fuse_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 415872; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) < 3192) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57) < 56)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3249) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3249) / 57)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 57))] : 0.000000e+00f);
  }
}

__kernel void fuse_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) < 812) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29) < 28)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 841) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 841) / 29)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 29))] : 0.000000e+00f);
  }
}

__kernel void fuse_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) < 210) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15) < 14)) ? input0[(((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 225) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 225) / 15)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 15))] : 0.000000e+00f);
  }
}

// avgpool, transpose, reshape/copy, FC, softmax
__kernel void fuse_global_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    tensor[ax1] = 0.000000e+00f;
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      for (int rv2 = 0; rv2 < 7; ++rv2) {
        tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv1) * 7) + rv2)] * 2.040816e-02f));
      }
    }
  }
}

__kernel void fuse_transpose_flatten_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1024; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_reshape[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_kernel1(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    float tmp = 0.000000e+00f;
    #pragma unroll 32
    for (int rc = 0; rc < 1024; ++rc) {
      tmp += input0[rc] * input1[((ax1 * 1024) + rc)];
    }
    T_add[ax1] = (tmp + input2[ax1]);
  }
}

__kernel void fuse_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1000; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
  }
}

__kernel void fuse_softmax_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
  float maxf = -3.402823e+38f;
  for (int k1 = 0; k1 < 1000; ++k1) {
    maxf = max(maxf, input0[k1]);
  }

  for (int ax1 = 0; ax1 < 1000; ++ax1) {
    float tmp = 0.000000e+00f;
    #pragma unroll 20
    for (int k2 = 0; k2 < 1000; ++k2) {
      tmp += exp((input0[k2] - maxf));
    }
    tensor2[ax1] = (exp((input0[ax1] - maxf)) / tmp);
  }
}

__kernel void fuse_softmax_ref(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
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


