#include "common.h"
#include "networks.h"

namespace config {

/* Modify this to choose network */
OctoCfg *octocfg = &LeNet5_Reuse;

OctoCfg LeNet5 {
    .f_weight="../data/mnist_weight_dump.txt",
    .f_bufsizes="",
    .cfg_network = LeNet5::base_network,
    .importer = Importer(10000, 784, "../data/mnist_test.db", "../data/mnist_test_y.db"),
    .executor = MNISTExecutor(10000, 10, MNISTExecutorType::BASE)
};

OctoCfg LeNet5_Reuse {
    .f_weight="../data/mnist_weight_dump.txt",
    .f_bufsizes="",
    .cfg_network = LeNet5::reuse_fused_network,
    .importer = Importer(10000, 784, "../data/mnist_test.db", "../data/mnist_test_y.db"),
    .executor = MNISTExecutor(10000, 10, MNISTExecutorType::REUSE)
};

OctoCfg MobileNetV2 {
    .f_weight="../data/inet_mnet_params.txt",
    .f_bufsizes="../data/inet_mnet_channels_bufsizes.txt",
    .cfg_network = ImageNet::MobileNet_channels,
    .importer = Importer(1, 224*224*3, "../data/cat244244.db", ""), 
    .executor = Executor(1, 1000)
};

OctoCfg SqueezeNet {
    .f_weight="../data/inet_sqnet_params.txt",
    .f_bufsizes="",
    .cfg_network = ImageNet::SqueezeNet,
    .importer = Importer(1, 224*224*3, "../data/cat244244.db", ""),
    .executor = Executor(1, 1000)
};

cfgs CfgList = {
    {"lenet5_baseline", LeNet5},
    {"lenet5_reuse", LeNet5_Reuse},
    {"im_mobilenetv2", MobileNetV2},
    {"im_squeezenet", SqueezeNet}
};

void printcfgs() {
    printf("Available configurations: \n");
    for (auto &i : CfgList) {
        printf("%s\n", i.first.c_str());
    }
}

OctoCfg* select_config(std::string cfgname) {
    for (auto &i : CfgList) {
        if (i.first == cfgname) return &(i.second);
    }
    return NULL;
}

}
