#ifndef LULESH_KINEMATICS_H
#define LULESH_KINEMATICS_H

#include "lulesh.h"

// 计算所有单元的运动学量（体积、速度梯度、变形率）
void CalcKinematicsForElems(Domain &domain, Real_t *vnew,
                            Real_t deltaTime, Index_t numElem);

// 单元 Lagrange 步骤：运动学计算并检查体积
void CalcLagrangeElements(Domain& domain, Real_t* vnew);

#endif // LULESH_KINEMATICS_H
