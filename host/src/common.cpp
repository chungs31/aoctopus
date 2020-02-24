#include "common.h"
#include "networks.h"
#include <cstdlib>

static std::string static_path(const std::string f) {
    return std::string(std::string(getenv("HOME")) + "/aoctopus_static/data/" + f);
}

namespace config {

/* Modify this to choose network */
OctoCfg *octocfg;

OctoCfg LeNet5 {
    .f_weight=static_path("mnist_weight_dump.txt"),
    .f_bufsizes="",
    .cfg_network = LeNet5::base_network,
    .importer = Importer(10000, 784, static_path("mnist_test.db"), static_path("mnist_test_y.db")),
    .executor = new MNISTExecutor(10000, 10, 0.98, MNISTExecutorType::BASE)
};

OctoCfg LeNet5_Unrolled {
    .f_weight=static_path("mnist_weight_dump.txt"),
    .f_bufsizes="",
    .cfg_network = LeNet5::base_network,
    .importer = Importer(10000, 784, static_path("mnist_test.db"), static_path("mnist_test_y.db")),
    .executor = new MNISTExecutor(10000, 10, 0.98, MNISTExecutorType::BASE)
};

OctoCfg LeNet5_Channels {
    .f_weight=static_path("mnist_weight_dump.txt"),
    .f_bufsizes="",
    .cfg_network = LeNet5::channels_network,
    .importer = Importer(10000, 784, static_path("mnist_test.db"), static_path("mnist_test_y.db")),
    .executor = new MNISTExecutor(10000, 10, 0.98, MNISTExecutorType::BASE)
};

OctoCfg LeNet5_Autorun {
    .f_weight=static_path("mnist_weight_dump.txt"),
    .f_bufsizes="",
    .cfg_network = LeNet5::autorun_network,
    .importer = Importer(10000, 784, static_path("mnist_test.db"), static_path("mnist_test_y.db")),
    .executor = new MNISTExecutor(10000, 10, 0.98, MNISTExecutorType::BASE)
};

OctoCfg LeNet5_Reuse {
    .f_weight=static_path("mnist_weight_dump.txt"),
    .f_bufsizes="",
    .cfg_network = LeNet5::reuse_fused_network,
    .importer = Importer(10000, 784, static_path("mnist_test.db"), static_path("mnist_test_y.db")),
    .executor = new MNISTExecutor(10000, 10, 0.98, MNISTExecutorType::REUSE)
};

OctoCfg MobileNetV2 {
    .f_weight=static_path("inet_mnet_params.txt"),
    .f_bufsizes=static_path("inet_mnet_bufsizes.txt"),
    .cfg_network = ImageNet::MobileNet,
    .importer = Importer(1, 224*224*3, static_path("cat224224.db"), static_path("cat224224_y.db")), 
    .executor = new MobileNetExecutor(1, 1000, 0.0, MobileNetExecutorType::BASE)
};

OctoCfg MobileNetV2_Channels {
    .f_weight=static_path("inet_mnet_params.txt"),
    .f_bufsizes=static_path("inet_mnet_channels_bufsizes.txt"),
    .cfg_network = ImageNet::MobileNet_channels,
    .importer = Importer(1, 224*224*3, static_path("cat224224.db"), static_path("cat224224_y.db")), 
    .executor = new MobileNetExecutor(1, 1000, 0.0, MobileNetExecutorType::CHANNELS)
};

/*
OctoCfg SqueezeNet {
    .f_weight="../data/inet_sqnet_params.txt",
    .f_bufsizes="",
    .cfg_network = ImageNet::SqueezeNet,
    .importer = Importer(1, 224*224*3, "../data/cat244244.db", ""),
    .executor = new ImageNetExecutor(1, 1000)
};
*/

cfgs CfgList = {
    {"lenet5_baseline", LeNet5},
    {"lenet5_unrolled", LeNet5_Unrolled},
    {"lenet5_channels", LeNet5_Channels},
    {"lenet5_autorun", LeNet5_Autorun},
    {"lenet5_reuse", LeNet5_Reuse},
    {"im_mnetv2_baseline", MobileNetV2},
    {"im_mnetv2_channels", MobileNetV2_Channels}
    //{"im_squeezenet", SqueezeNet}
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
