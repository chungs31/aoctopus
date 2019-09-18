#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

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


