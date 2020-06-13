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
#include <utility>
#include <vector>

/* Forward declaration */
class Layer;

/* External shared variables
 */
extern int TEST_SET_SIZE;
extern bool TEST_RANDOM_INPUT;
extern cl_ulong kernel_time, write_time, read_time;

/* Configuration
 */
namespace config {

struct OctoCfg {
    std::string f_weight;
    std::string f_bufsizes;
    Layer *cfg_network;
    Importer importer;
    Executor *executor;
};

typedef std::pair<std::string, OctoCfg> cfgpair;
typedef std::vector<cfgpair> cfgs;
extern cfgs CfgList;

extern OctoCfg LeNet5;
extern OctoCfg LeNet5_Unrolled;
extern OctoCfg LeNet5_Channels;
extern OctoCfg LeNet5_Autorun;
extern OctoCfg LeNet5_Reuse;

extern OctoCfg MobileNetV2;
extern OctoCfg SqueezeNet;

extern OctoCfg *octocfg;

void printcfgs();
OctoCfg* select_config(std::string cfgname);

}

#endif /* COMMON_H */

