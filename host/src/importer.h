/* 2019-09-17 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * importer.h
 *
 * Text file parser for importing weights trained in Keras.
 */


#ifndef IMPORTER_H
#define IMPORTER_H

#include <vector>
#include "AOCLUtils/aocl_utils.h"

void weight_parser(const char *filename, std::vector<std::vector<float> > &weights);
void bufsizes_parser(const char *filename, std::vector<std::vector<size_t> > &weights);
void import_mnist(const char *x_test, const char *y_test, aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &mnist_x_test, aocl_utils::scoped_array<int> &mnist_y_test);
void import_imagenet(const char *x_test, const char *y_test, aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &out_x_test, aocl_utils::scoped_array<int> &out_y_test);
void generate_random(int num_inputs, int input_size, aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &x_test);

#endif
