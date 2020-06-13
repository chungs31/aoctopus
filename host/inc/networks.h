/* Network declaration file */

#ifndef NETWORKS_H
#define NETWORKS_H

#include "layer.h"

namespace LeNet5 {

extern Layer base_network[]; // Default network (Baseline, Unrolled)
extern Layer channels_network[];
extern Layer autorun_network[];
extern Layer reuse_network[];
extern Layer reuse_fused_network[];
extern Layer combined_dense_network[];
extern Layer globalmem_network[];

}

namespace ImageNet {

extern Layer SqueezeNet[];
extern Layer MobileNet[];
extern Layer MobileNet_isolate[];
extern Layer MobileNet_channels[];
extern Layer MobileNet_opt[];

}

#endif

