
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float ch0 __attribute__((depth(4056)));
channel float ch1 __attribute__((depth(1014)));
channel float ch2 __attribute__((depth(1936)));
channel float ch3 __attribute__((depth(400)));
channel float ch4 __attribute__((depth(400)));
channel float ch5 __attribute__((depth(120)));
channel float ch6 __attribute__((depth(84)));
channel float ch7 __attribute__((depth(10)));

__kernel void fused_nn_conv2d_relu_1_kernel0(__global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  #pragma unroll 1
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 26; ++ax2) {
      float compute[1];
      for (int ax3_outer = 0; ax3_outer < 26; ++ax3_outer) {
        compute[(0)] = 0.000000e+00f;
        compute[(0)] = (compute[(0)] + (placeholder[(((ax2 * 28) + ax3_outer))] * placeholder1[((ax1 * 9))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 1))] * placeholder1[(((ax1 * 9) + 1))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 2))] * placeholder1[(((ax1 * 9) + 2))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 28))] * placeholder1[(((ax1 * 9) + 3))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 29))] * placeholder1[(((ax1 * 9) + 4))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 30))] * placeholder1[(((ax1 * 9) + 5))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 56))] * placeholder1[(((ax1 * 9) + 6))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 57))] * placeholder1[(((ax1 * 9) + 7))]));
        compute[(0)] = (compute[(0)] + (placeholder[((((ax2 * 28) + ax3_outer) + 58))] * placeholder1[(((ax1 * 9) + 8))]));
        //T_relu[((((ax1 * 676) + (ax2 * 26)) + ax3_outer))] = max((compute[(0)] + placeholder2[(ax1)]), 0.000000e+00f);
        write_channel_intel(ch0, max((compute[(0)] + placeholder2[(ax1)]), 0.000000e+00f));
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fused_nn_avg_pool2d_1_kernel0() {
  float in[4056];
  for (int i = 0; i < 4056; ++i) {
    in[i] = read_channel_intel(ch0);
  }
  
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    float tensor1[1];
    for (int ax2 = 0; ax2 < 13; ++ax2) {
      for (int ax3 = 0; ax3 < 13; ++ax3) {
        tensor1[(0)] = 0.000000e+00f;
        tensor1[(0)] = (tensor1[(0)] + in[((((ax1 * 676) + (ax2 * 52)) + (ax3 * 2)))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 676) + (ax2 * 52)) + (ax3 * 2)) + 1))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 676) + (ax2 * 52)) + (ax3 * 2)) + 26))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 676) + (ax2 * 52)) + (ax3 * 2)) + 27))]);
        //tensor[((((ax1 * 169) + (ax2 * 13)) + ax3))] = (tensor1[(0)] * 2.500000e-01f);
        write_channel_intel(ch1, (tensor1[(0)] * 2.500000e-01f));
      }
    }
  }
}

__kernel void fused_nn_conv2d_relu_kernel0(__global float* restrict placeholder1, __global float* restrict placeholder2) {
  float in[1014];
  for (int i = 0; i < 1014; i++) {
    in[i] = read_channel_intel(ch1);
  }
  
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 11; ++ax2) {
      float compute[1];
      for (int ax3_outer = 0; ax3_outer < 11; ++ax3_outer) {
        compute[(0)] = 0.000000e+00f;
        compute[(0)] = (compute[(0)] + (in[(((ax2 * 13) + ax3_outer))] * placeholder1[((ax1 * 54))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 1))] * placeholder1[(((ax1 * 54) + 1))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 2))] * placeholder1[(((ax1 * 54) + 2))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 13))] * placeholder1[(((ax1 * 54) + 3))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 14))] * placeholder1[(((ax1 * 54) + 4))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 15))] * placeholder1[(((ax1 * 54) + 5))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 26))] * placeholder1[(((ax1 * 54) + 6))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 27))] * placeholder1[(((ax1 * 54) + 7))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 28))] * placeholder1[(((ax1 * 54) + 8))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 169))] * placeholder1[(((ax1 * 54) + 9))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 170))] * placeholder1[(((ax1 * 54) + 10))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 171))] * placeholder1[(((ax1 * 54) + 11))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 182))] * placeholder1[(((ax1 * 54) + 12))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 183))] * placeholder1[(((ax1 * 54) + 13))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 184))] * placeholder1[(((ax1 * 54) + 14))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 195))] * placeholder1[(((ax1 * 54) + 15))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 196))] * placeholder1[(((ax1 * 54) + 16))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 197))] * placeholder1[(((ax1 * 54) + 17))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 338))] * placeholder1[(((ax1 * 54) + 18))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 339))] * placeholder1[(((ax1 * 54) + 19))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 340))] * placeholder1[(((ax1 * 54) + 20))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 351))] * placeholder1[(((ax1 * 54) + 21))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 352))] * placeholder1[(((ax1 * 54) + 22))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 353))] * placeholder1[(((ax1 * 54) + 23))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 364))] * placeholder1[(((ax1 * 54) + 24))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 365))] * placeholder1[(((ax1 * 54) + 25))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 366))] * placeholder1[(((ax1 * 54) + 26))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 507))] * placeholder1[(((ax1 * 54) + 27))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 508))] * placeholder1[(((ax1 * 54) + 28))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 509))] * placeholder1[(((ax1 * 54) + 29))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 520))] * placeholder1[(((ax1 * 54) + 30))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 521))] * placeholder1[(((ax1 * 54) + 31))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 522))] * placeholder1[(((ax1 * 54) + 32))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 533))] * placeholder1[(((ax1 * 54) + 33))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 534))] * placeholder1[(((ax1 * 54) + 34))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 535))] * placeholder1[(((ax1 * 54) + 35))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 676))] * placeholder1[(((ax1 * 54) + 36))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 677))] * placeholder1[(((ax1 * 54) + 37))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 678))] * placeholder1[(((ax1 * 54) + 38))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 689))] * placeholder1[(((ax1 * 54) + 39))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 690))] * placeholder1[(((ax1 * 54) + 40))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 691))] * placeholder1[(((ax1 * 54) + 41))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 702))] * placeholder1[(((ax1 * 54) + 42))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 703))] * placeholder1[(((ax1 * 54) + 43))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 704))] * placeholder1[(((ax1 * 54) + 44))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 845))] * placeholder1[(((ax1 * 54) + 45))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 846))] * placeholder1[(((ax1 * 54) + 46))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 847))] * placeholder1[(((ax1 * 54) + 47))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 858))] * placeholder1[(((ax1 * 54) + 48))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 859))] * placeholder1[(((ax1 * 54) + 49))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 860))] * placeholder1[(((ax1 * 54) + 50))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 871))] * placeholder1[(((ax1 * 54) + 51))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 872))] * placeholder1[(((ax1 * 54) + 52))]));
        compute[(0)] = (compute[(0)] + (in[((((ax2 * 13) + ax3_outer) + 873))] * placeholder1[(((ax1 * 54) + 53))]));
        //T_relu[((((ax1 * 121) + (ax2 * 11)) + ax3_outer))] = max((compute[(0)] + placeholder2[(ax1)]), 0.000000e+00f);
        write_channel_intel(ch2, max((compute[(0)] + placeholder2[(ax1)]), 0.000000e+00f));
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fused_nn_avg_pool2d_kernel0() {
  float in[1936]; 
  for (int i = 0; i < 1936; i++){ 
    in[i] = read_channel_intel(ch2);
  }
  
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    float tensor1[1];
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        tensor1[(0)] = 0.000000e+00f;
        tensor1[(0)] = (tensor1[(0)] + in[((((ax1 * 121) + (ax2 * 22)) + (ax3 * 2)))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 121) + (ax2 * 22)) + (ax3 * 2)) + 1))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 121) + (ax2 * 22)) + (ax3 * 2)) + 11))]);
        tensor1[(0)] = (tensor1[(0)] + in[(((((ax1 * 121) + (ax2 * 22)) + (ax3 * 2)) + 12))]);
        //tensor[((((ax1 * 25) + (ax2 * 5)) + ax3))] = (tensor1[(0)] * 2.500000e-01f);
        write_channel_intel(ch3, (tensor1[(0)] * 2.500000e-01f));
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void fused_transpose_nn_batch_flatten_kernel0() {
  float in[400];
  for (int i = 0; i < 400; i++) {
    in[i] = read_channel_intel(ch3);
  }
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 400; ++ax0_ax1_fused_inner) {
    //tensor[(ax0_ax1_fused_inner)] = placeholder[((((ax0_ax1_fused_inner & 15) * 25) + (ax0_ax1_fused_inner >> 4)))];
    write_channel_intel(ch4, in[((((ax0_ax1_fused_inner & 15) * 25) + (ax0_ax1_fused_inner >> 4)))]);
  }
}

__kernel void fused_nn_dense_relu_1_kernel0(__global float* restrict placeholder1, __global float* restrict placeholder2) {
  float in[400];
  for (int i = 0; i < 400; i++){
    in[i] = read_channel_intel(ch4);
  }
  
  for (int ax1 = 0; ax1 < 120; ++ax1) {
    float T_dense[1];
    T_dense[(0)] = 0.000000e+00f;
    for (int k_outer = 0; k_outer < 10; ++k_outer) {
      T_dense[(0)] = (T_dense[(0)] + (in[((k_outer * 40))] * placeholder1[(((ax1 * 400) + (k_outer * 40)))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 1))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 1))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 2))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 2))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 3))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 3))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 4))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 4))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 5))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 5))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 6))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 6))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 7))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 7))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 8))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 8))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 9))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 9))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 10))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 10))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 11))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 11))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 12))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 12))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 13))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 13))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 14))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 14))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 15))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 15))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 16))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 16))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 17))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 17))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 18))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 18))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 19))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 19))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 20))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 20))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 21))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 21))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 22))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 22))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 23))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 23))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 24))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 24))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 25))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 25))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 26))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 26))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 27))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 27))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 28))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 28))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 29))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 29))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 30))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 30))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 31))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 31))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 32))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 32))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 33))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 33))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 34))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 34))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 35))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 35))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 36))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 36))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 37))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 37))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 38))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 38))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 39))] * placeholder1[((((ax1 * 400) + (k_outer * 40)) + 39))]));
    }
    //T_relu[(ax1)] = max((T_dense[(0)] + placeholder2[(ax1)]), 0.000000e+00f);
    write_channel_intel(ch5, max((T_dense[(0)] + placeholder2[(ax1)]), 0.000000e+00f));
  }
}

__kernel void fused_nn_dense_relu_kernel0(__global float* restrict placeholder1, __global float* restrict placeholder2) {
  float in[120];
  for (int i = 0; i < 120; i++) {
    in[i] = read_channel_intel(ch5);
  }
  for (int ax1 = 0; ax1 < 84; ++ax1) {
    float T_dense[1];
    T_dense[(0)] = 0.000000e+00f;
    for (int k_outer = 0; k_outer < 3; ++k_outer) {
      T_dense[(0)] = (T_dense[(0)] + (in[((k_outer * 40))] * placeholder1[(((ax1 * 120) + (k_outer * 40)))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 1))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 1))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 2))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 2))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 3))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 3))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 4))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 4))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 5))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 5))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 6))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 6))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 7))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 7))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 8))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 8))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 9))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 9))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 10))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 10))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 11))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 11))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 12))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 12))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 13))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 13))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 14))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 14))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 15))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 15))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 16))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 16))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 17))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 17))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 18))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 18))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 19))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 19))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 20))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 20))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 21))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 21))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 22))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 22))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 23))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 23))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 24))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 24))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 25))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 25))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 26))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 26))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 27))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 27))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 28))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 28))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 29))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 29))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 30))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 30))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 31))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 31))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 32))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 32))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 33))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 33))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 34))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 34))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 35))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 35))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 36))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 36))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 37))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 37))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 38))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 38))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 40) + 39))] * placeholder1[((((ax1 * 120) + (k_outer * 40)) + 39))]));
    }
    //T_relu[(ax1)] = max((T_dense[(0)] + placeholder2[(ax1)]), 0.000000e+00f);
    write_channel_intel(ch6, max((T_dense[(0)] + placeholder2[(ax1)]), 0.000000e+00f));
  }
}

__kernel void fused_nn_dense_nn_bias_add_kernel0(__global float* restrict placeholder1, __global float* restrict placeholder2) {
  float in[84];
  for (int i = 0; i < 84; i++) {
    in[i] = read_channel_intel(ch6);
  }
  
  for (int ax1 = 0; ax1 < 10; ++ax1) {
    float T_dense[1];
    T_dense[(0)] = 0.000000e+00f;
    for (int k_outer = 0; k_outer < 21; ++k_outer) {
      T_dense[(0)] = (T_dense[(0)] + (in[((k_outer * 4))] * placeholder1[(((ax1 * 84) + (k_outer * 4)))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 4) + 1))] * placeholder1[((((ax1 * 84) + (k_outer * 4)) + 1))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 4) + 2))] * placeholder1[((((ax1 * 84) + (k_outer * 4)) + 2))]));
      T_dense[(0)] = (T_dense[(0)] + (in[(((k_outer * 4) + 3))] * placeholder1[((((ax1 * 84) + (k_outer * 4)) + 3))]));
    }
    //T_add[(ax1)] = (T_dense[(0)] + placeholder2[(ax1)]);
    write_channel_intel(ch7, (T_dense[(0)] + placeholder2[(ax1)]));
  }
}

__kernel void fused_nn_softmax_kernel0(__global float* restrict T_softmax_norm) {
  float in[10];
  float maxelem = -3.402823e+38f;
  float exps[10];
  float expsum = 0.000000e+00f;
  for (int i = 0; i < 10; i++) {
    in[i] = read_channel_intel(ch7);
  }
  for (int k = 0; k < 10; ++k) {
    maxelem = max(maxelem, in[k]);
  }
  for (int i1 = 0; i1 < 10; ++i1) {
    exps[(i1)] = exp((in[(i1)] - maxelem));
  }
  for (int k1 = 0; k1 < 10; ++k1) {
    expsum = (expsum + exps[(k1)]);
  }
  for (int i11 = 0; i11 < 10; ++i11) {
    T_softmax_norm[(i11)] = (exps[(i11)] / expsum);
  }
}
