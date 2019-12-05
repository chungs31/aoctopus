/* 2019-12-04 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * test_pcie_bandwidth.h
 *
 * Test the performance of CL buffer copy functions. 
 */

#ifndef TEST_PCIE_BANDWIDTH_H
#define TEST_PCIE_BANDWIDTH_H

void pcie_bandwidth_test();
bool init_opencl();
void cleanup();
float rand_float();

#endif

