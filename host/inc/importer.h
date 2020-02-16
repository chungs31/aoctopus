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
#include <string>
#include "AOCLUtils/aocl_utils.h"

class Importer {
public:
    int num_inputs;
    int input_dim;
    std::string f_input;
    std::string f_ref_output;

    Importer() {};
    Importer(int num, int dim, std::string f_i, std::string f_o) : num_inputs(num), input_dim(dim), f_input(f_i), f_ref_output(f_o) {};

    void import_input_data(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &x_test, aocl_utils::scoped_array<int> &y_test);
    void generate_random_input(int num_inputs, aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &x_test); 
};

void weight_parser(const char *filename, std::vector<std::vector<float> > &weights);
void bufsizes_parser(const char *filename, std::vector<std::vector<size_t> > &weights);

#endif
