#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AOCLUtils/aocl_utils.h"
#include "importer.h"
#include <assert.h>
#include <string.h>

using namespace std;
using namespace aocl_utils;

static float rand_float();
template <class T>
static void import_input_dataset(
        unsigned int num_inputs,
        unsigned int input_dim,
        const char *x_test_path, 
        const char *y_test_path, 
        scoped_array<scoped_aligned_ptr<T> > &x_test, 
        scoped_array<int> &y_test
);

/* Parse weights from filename and put it into a vector (layer i) of vector of floats. */
void weight_parser(const char *filename, vector<vector<float> > &weights) {
    if (strcmp("", filename) == 0) return;

    ifstream infile(filename);
    try {
        infile.exceptions(ifstream::failbit | ifstream::badbit);

        string line;
        while (getline(infile, line)) {
            int num = 0;
            stringstream ss(line);
            float weight;
            char dummy;
            ss >> dummy;
            vector<float> curr_weights;

            while (ss >> weight) {
                curr_weights.push_back(weight);
                num++;
                ss >> dummy;
            }
            weights.push_back(curr_weights);
        }

        infile.close();
    }
    catch (ifstream::failure &e) {
        if (!infile.eof()) {
            cerr << "[ERROR] could not read weight file at " << filename << "!\n";
            assert(0);
        }
    }
}

void bufsizes_parser(const char *filename, vector<vector<size_t> > &weights) {
    if (strcmp("", filename) == 0) return;

    ifstream infile(filename);
    try {
        infile.exceptions(ifstream::failbit | ifstream::badbit);

        size_t sum = 0;
        string line;
        while (getline(infile, line)) {
            int num = 0;
            stringstream ss(line);
            size_t weight;
            char dummy;
            do {
                ss >> dummy;
            } while (dummy != ',');
            vector<size_t> curr_weights;

            while (ss >> weight) {
                sum += weight;
                weight /= sizeof(float);
                curr_weights.push_back(weight);
                num++;
                ss >> dummy;
            }
            weights.push_back(curr_weights);
        }

        infile.close();
    }
    catch (ifstream::failure &e) {
        if (!infile.eof()) {
            cerr << "[ERROR] could not read bufsizes file at " << filename << "!\n";
            assert(0);
        }
    }
}

void Importer::import_input_data(
        scoped_array<scoped_aligned_ptr<float>> &x_test, 
        scoped_array<int> &y_test
) {
    import_input_dataset(num_inputs, input_dim, f_input.c_str(), f_ref_output.c_str(), x_test, y_test); 
}

void Importer::generate_random_input(int _num_inputs, scoped_array<scoped_aligned_ptr<float> > &x_test) {
    num_inputs = _num_inputs;
    x_test.reset(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        x_test[i].reset(input_dim);
        for (int j = 0; j < input_dim; j++) {
            x_test[i][j] = rand_float();
        }
    }
}

static float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

template <class T>
static void import_input_dataset(
        unsigned int num_inputs,
        unsigned int input_dim,
        const char *x_test_path, 
        const char *y_test_path, 
        scoped_array<scoped_aligned_ptr<T> > &x_test, 
        scoped_array<int> &y_test
) {
    ifstream infile(x_test_path);
    try {
        infile.exceptions(ifstream::failbit | ifstream::badbit);
        
        /* Obtain input data */
        x_test.reset(num_inputs);
        string line;
        int i = 0;
        while (getline(infile, line)) {
            x_test[i].reset(input_dim);
            stringstream ss(line);
            T weight;
            char dummy;
            ss >> dummy;
            int j = 0;

            while (ss >> weight) {
                x_test[i][j] = weight;
                j++;
                ss >> dummy;
            }
            i++;
        }
        infile.close();
    }
    catch (ifstream::failure &e) {
        if (!infile.eof()) {
            cerr << "[ERROR] could not read input file at " << x_test_path << "!\n";
            assert(0);
        }
    }

    // Ignore if reference answer path not given
    if (strcmp(y_test_path, "") == 0) return;

    ifstream yfile(y_test_path);
    try {
        yfile.exceptions(ifstream::failbit | ifstream::badbit);
        /* Get reference answers */
        y_test.reset(num_inputs);
        
        string line;
        getline(yfile, line);
        stringstream ss(line);
        int y; // Assuming classification is int (index)
        int i = 0;
        char dummy;
        ss >> dummy;
        while (ss >> y) {
            y_test[i++] = y;
            ss >> dummy;
        }
        yfile.close();
    }
    catch (ifstream::failure &e) {
        if (!yfile.eof()) {
            cerr << "[ERROR] could not read reference input file at " << y_test_path << "!\n";
            assert(0);
        }
    }
}

