#ifndef LULESH_GEOMETRY_H
#define LULESH_GEOMETRY_H

#include "lulesh.h"

// 从 Domain 中提取单元的8个节点坐标
void CollectDomainNodesToElemNodes(Domain &domain,
                                   const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8]);

// 计算形函数导数（B矩阵）和雅可比行列式（体积）
void CalcElemShapeFunctionDerivatives(Real_t const x[],
                                      Real_t const y[],
                                      Real_t const z[],
                                      Real_t b[][8],
                                      Real_t* const volume);

// 计算单元各节点的法向量
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8]);

// 计算单元体积对节点坐标的导数（8个节点各3个分量）
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8]);

// 计算六面体单元体积（24参数版本）
Real_t CalcElemVolume(const Real_t x0, const Real_t x1,
                      const Real_t x2, const Real_t x3,
                      const Real_t x4, const Real_t x5,
                      const Real_t x6, const Real_t x7,
                      const Real_t y0, const Real_t y1,
                      const Real_t y2, const Real_t y3,
                      const Real_t y4, const Real_t y5,
                      const Real_t y6, const Real_t y7,
                      const Real_t z0, const Real_t z1,
                      const Real_t z2, const Real_t z3,
                      const Real_t z4, const Real_t z5,
                      const Real_t z6, const Real_t z7);

// 计算六面体单元体积（数组接口）
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8]);

// 计算单元特征长度（用于时间步约束）
Real_t CalcElemCharacteristicLength(const Real_t x[8],
                                    const Real_t y[8],
                                    const Real_t z[8],
                                    const Real_t volume);

#endif // LULESH_GEOMETRY_H
