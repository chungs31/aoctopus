#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AOCLUtils/aocl_utils.h"

using namespace std;
using namespace aocl_utils;

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


