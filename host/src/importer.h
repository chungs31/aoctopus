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

class Importer {
public:
    int num_inputs;
    int input_dim;

    Importer() {};
    Importer(int num, int dim) : num_inputs(num), input_dim(dim) {};

    virtual void import_input_data(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &x_test, aocl_utils::scoped_array<int> &y_test) = 0;
    void generate_random_input(int num_inputs, aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float> > &x_test); 
};

class MNIST_Importer : public Importer {
public:
    MNIST_Importer() : Importer{10000, 784} {};
    virtual void import_input_data(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &x_test, aocl_utils::scoped_array<int> &y_test) override;
};

class ImageNet_Importer : public Importer {
public:
    ImageNet_Importer() : Importer{1, 224*224*3} {};
    virtual void import_input_data(aocl_utils::scoped_array<aocl_utils::scoped_aligned_ptr<float>> &x_test, aocl_utils::scoped_array<int> &y_test) override;
};

void weight_parser(const char *filename, std::vector<std::vector<float> > &weights);
void bufsizes_parser(const char *filename, std::vector<std::vector<size_t> > &weights);

#endif
