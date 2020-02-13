#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AOCLUtils/aocl_utils.h"
#include "importer.h"

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
    ifstream infile(filename);

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
        //cout << num << endl;
    }

    infile.close();
}

void bufsizes_parser(const char *filename, vector<vector<size_t> > &weights) {
    ifstream infile(filename);

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
        //ss >> dummy;
        vector<size_t> curr_weights;

        while (ss >> weight) {
            sum += weight;
            weight /= sizeof(float);
            curr_weights.push_back(weight);
            num++;
            ss >> dummy;
        }
        weights.push_back(curr_weights);
        //cout << num << endl;
    }

    std::cout << "Total bytes from bufsizes: " << sum << std::endl;

    infile.close();
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
    ifstream yfile(y_test_path);
    
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

    /* Get reference answers */
    y_test.reset(num_inputs);
    
    getline(yfile, line);
    stringstream ss(line);
    int y; // Assuming classification is int (index)
    i = 0;
    char dummy;
    ss >> dummy;
    while (ss >> y) {
        y_test[i++] = y;
        ss >> dummy;
    }

    infile.close();
    yfile.close();
}

