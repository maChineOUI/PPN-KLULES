#ifndef LULESH_NODAL_H
#define LULESH_NODAL_H

#include "lulesh.h"

// 完整的节点 Lagrange 步骤：力 → 加速度 → 速度 → 位置
void LagrangeNodal(Domain& domain);

#endif // LULESH_NODAL_H
