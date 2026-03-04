#pragma once

#include "lulesh.h"

// 单元 Lagrange 步骤：运动学计算（已与 CalcKinematicsForElems 融合）
void CalcLagrangeElements(Domain& domain, Real_t* vnew);
