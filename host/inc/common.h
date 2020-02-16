/* 2019-10-01 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * common.h
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include "CL/opencl.h"
#include "importer.h"
#include "runtime.h"
#include <string>

class Layer;


/* External shared variables
 */
extern int TEST_SET_SIZE;
extern cl_ulong kernel_time, write_time, read_time;

/* Configuration
 */
namespace config {

struct OctoCfg {
    std::string f_weight;
    std::string f_bufsizes;
    Layer *cfg_network;
    Importer importer;
    Executor executor;
};

extern OctoCfg LeNet5;
extern OctoCfg MobileNetV2;
extern OctoCfg SqueezeNet;

extern OctoCfg *octocfg;

}

#endif /* COMMON_H */

