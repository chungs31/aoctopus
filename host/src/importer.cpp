#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AOCLUtils/aocl_utils.h"

using namespace std;
using namespace aocl_utils;

float rand_float(); 

float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

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

void import_mnist(const char *x_test, const char *y_test, aocl_utils::scoped_array<scoped_aligned_ptr<float> > &mnist_x_test, aocl_utils::scoped_array<int> &mnist_y_test) {
    ifstream infile(x_test);
    mnist_x_test.reset(10000);

    string line;
    int i = 0;
    while (getline(infile, line)) {
        mnist_x_test[i].reset(784);
        stringstream ss(line);
        float weight;
        char dummy;
        ss >> dummy;
        int j = 0;

        while (ss >> weight) {
            mnist_x_test[i][j] = weight;
            j++;
            ss >> dummy;
        }
        i++;
    }


    ifstream yfile(y_test);
    mnist_y_test.reset(10000);
    
    getline(yfile, line);
    stringstream ss(line);
    int y;
    i = 0;
    char dummy;
    ss >> dummy;
    while (ss >> y) {
        mnist_y_test[i++] = y;
        ss >> dummy;
    }

    infile.close();
    yfile.close();
}

void import_imagenet(const char *x_test, const char *y_test, aocl_utils::scoped_array<scoped_aligned_ptr<float> > &out_x_test, aocl_utils::scoped_array<int> &out_y_test) {
    ifstream infile(x_test);
    out_x_test.reset(1);

    string line;
    int i = 0;
    while (getline(infile, line)) {
        out_x_test[i].reset(224*224*3);
        stringstream ss(line);
        float weight;
        char dummy;
        ss >> dummy;
        int j = 0;

        while (ss >> weight) {
            out_x_test[i][j] = weight;
            j++;
            ss >> dummy;
        }
        i++;
    }


    /*
    ifstream yfile(y_test);
    out_y_test.reset(1);
    
    getline(yfile, line);
    stringstream ss(line);
    int y;
    i = 0;
    char dummy;
    ss >> dummy;
    while (ss >> y) {
        out_y_test[i++] = y;
        ss >> dummy;
    }
    */

    infile.close();
    //yfile.close();
}

void generate_random(int num_inputs, int input_size, aocl_utils::scoped_array<scoped_aligned_ptr<float> > &x_test) {
    x_test.reset(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        x_test[i].reset(input_size);
        for (int j = 0; j < input_size; j++) {
            x_test[i][j] = rand_float();
        }
    }
}

/*
int main() {
    vector<vector<float>> weights;

    weight_parser("mnist_weight_dump.txt", weights);

    for (auto i : weights) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }

    return 0;
}
*/


