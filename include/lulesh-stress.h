#ifndef LULESH_STRESS_H
#define LULESH_STRESS_H

#include "lulesh.h"

// 计算体积力（应力积分 + 沙漏控制），结果累积到节点力
void CalcVolumeForceForElems(Domain& domain);

#endif // LULESH_STRESS_H
