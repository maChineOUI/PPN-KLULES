#include <climits>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>

// Inclusion de Kokkos (version 5.0) / 引入 Kokkos 5.0
// 注意：Kokkos_Vector.hpp 已经不推荐使用，但为了保持与现有结构兼容暂保留

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>
#include <cstdio>

#if _OPENMP
#include <omp.h>
#endif

#include "lulesh.h"


// Allocation de tableaux via malloc（LULESH historique）
// 使用 malloc 的传统数组分配方式（注意：不用于 Kokkos::View，只用于 MPI 缓冲等）
template <typename T>
T* Allocate(size_t size) {
  return static_cast<T*>(malloc(sizeof(T) * size));
}

template <typename T>
void Release(T** ptr) {
  if (*ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
}


// TimeIncrement : mise à jour du pas de temps
// TimeIncrement：更新时间步（与原 LULESH 行为完全一致）
static inline void TimeIncrement(Domain& domain)
{
  // Temps restant avant l'arrêt / 剩余可用时间
  Real_t targetdt = domain.stoptime() - domain.time();

  // Mise à jour du pas de temps sauf si pas fixe ou cycle = 0
  // 若不是固定时间步且不是第一次迭代，则更新时间步
  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {

    Real_t ratio;
    Real_t olddt = domain.deltatime();

    Real_t gnewdt = Real_t(1.0e+20);   // très grand dt / 初始极大 dt
    Real_t newdt;

    // Limitation par dtcourant / Courant 条件
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0);
    }

    // Limitation par dthydro / 体积变化限制
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0);
    }

    // Réduction MPI si nécessaire / 若 MPI 启用则进行全域最小归约
#if USE_MPI
    MPI_Allreduce(&gnewdt, &newdt, 1,
                  ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                  MPI_MIN, MPI_COMM_WORLD);
#else
    newdt = gnewdt;
#endif

    // Ratio = nouveau / ancien，用于限制 dt 增长
    ratio = newdt / olddt;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt;
      }
      else if (ratio > domain.deltatimemultub()) {
        newdt = olddt * domain.deltatimemultub();
      }
    }

    // Ne jamais dépasser dtmax / 禁止超过 dtmax
    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax();
    }

    domain.deltatime() = newdt;
  }

  // Ajustement fin de step / 最后一阶段时间步调整
  if ((targetdt > domain.deltatime()) &&
      (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0)))) {

    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0);
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt;
  }

  // Mise à jour du temps / 更新时间
  domain.time() += domain.deltatime();

  // Incrément du cycle / 增加迭代次数
  ++domain.cycle();
}

// CollectDomainNodesToElemNodes : copie des coordonnées des nœuds d’élément
// 将单元的 8 个节点的坐标收集到 elemX/elemY/elemZ
//
// 注意：本函数在 KOKKOS_INLINE_FUNCTION 中运行
//      域访问必须使用 domain.x(i) 风格，确保 GPU 兼容性
KOKKOS_INLINE_FUNCTION
void CollectDomainNodesToElemNodes(const Domain& domain,
                                   const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
  Index_t nd0i = elemToNode[0];
  Index_t nd1i = elemToNode[1];
  Index_t nd2i = elemToNode[2];
  Index_t nd3i = elemToNode[3];
  Index_t nd4i = elemToNode[4];
  Index_t nd5i = elemToNode[5];
  Index_t nd6i = elemToNode[6];
  Index_t nd7i = elemToNode[7];

  elemX[0] = domain.x(nd0i);
  elemX[1] = domain.x(nd1i);
  elemX[2] = domain.x(nd2i);
  elemX[3] = domain.x(nd3i);
  elemX[4] = domain.x(nd4i);
  elemX[5] = domain.x(nd5i);
  elemX[6] = domain.x(nd6i);
  elemX[7] = domain.x(nd7i);

  elemY[0] = domain.y(nd0i);
  elemY[1] = domain.y(nd1i);
  elemY[2] = domain.y(nd2i);
  elemY[3] = domain.y(nd3i);
  elemY[4] = domain.y(nd4i);
  elemY[5] = domain.y(nd5i);
  elemY[6] = domain.y(nd6i);
  elemY[7] = domain.y(nd7i);

  elemZ[0] = domain.z(nd0i);
  elemZ[1] = domain.z(nd1i);
  elemZ[2] = domain.z(nd2i);
  elemZ[3] = domain.z(nd3i);
  elemZ[4] = domain.z(nd4i);
  elemZ[5] = domain.z(nd5i);
  elemZ[6] = domain.z(nd6i);
  elemZ[7] = domain.z(nd7i);
}

// InitStressTermsForElems : initialise les contraintes σxx σyy σzz
// 初始化单元的应力项 sigxx / sigyy / sigzz
//
// Dans LULESH original, sigxx/sigyy/sigzz sont des pointeurs classiques.
// L'utilisation de Kokkos ici est correcte car l’écriture se fait dans une
// boucle parallel_for simple et indépendante.
static inline void InitStressTermsForElems(Domain& domain,
                                           Real_t* sigxx,
                                           Real_t* sigyy,
                                           Real_t* sigzz,
                                           Index_t numElem)
{
  Kokkos::parallel_for(
    "InitStressTermsForElems",
    numElem,
    KOKKOS_LAMBDA(const Index_t& i) {
      // σ = -(p + q)
      // 三个方向的应力都等于 -(压力 + 人工粘性项)
      sigxx[i] = sigyy[i] = sigzz[i] = -domain.p(i) - domain.q(i);
    });
}

// CalcElemShapeFunctionDerivatives
// 计算形函数导数与单元体积（8 点六面体）
// 
// 注意：此函数是“数学几何计算”，不包含 Kokkos 并行语义，因此保持 CPU-only。
// 若未来需要在 GPU 上调用，可包装为 KOKKOS_FUNCTION。
static inline void CalcElemShapeFunctionDerivatives(Real_t const x[],
                                                    Real_t const y[],
                                                    Real_t const z[],
                                                    Real_t b[][8],
                                                    Real_t* const volume)
{
  // Coordonnées des 8 nœuds de l’élément
  // 8 个节点坐标（保持原始展开形式）
  const Real_t x0 = x[0];
  const Real_t x1 = x[1];
  const Real_t x2 = x[2];
  const Real_t x3 = x[3];
  const Real_t x4 = x[4];
  const Real_t x5 = x[5];
  const Real_t x6 = x[6];
  const Real_t x7 = x[7];

  const Real_t y0 = y[0];
  const Real_t y1 = y[1];
  const Real_t y2 = y[2];
  const Real_t y3 = y[3];
  const Real_t y4 = y[4];
  const Real_t y5 = y[5];
  const Real_t y6 = y[6];
  const Real_t y7 = y[7];

  const Real_t z0 = z[0];
  const Real_t z1 = z[1];
  const Real_t z2 = z[2];
  const Real_t z3 = z[3];
  const Real_t z4 = z[4];
  const Real_t z5 = z[5];
  const Real_t z6 = z[6];
  const Real_t z7 = z[7];

  // Variables intermédiaires
  // 中间导数项
  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  // Formules géométriques du Jacobien
  // Jacobian 几何公式（六面体标准推导）
  fjxxi = Real_t(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
  fjxet = Real_t(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
  fjxze = Real_t(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

  fjyxi = Real_t(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
  fjyet = Real_t(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
  fjyze = Real_t(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

  fjzxi = Real_t(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
  fjzet = Real_t(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
  fjzze = Real_t(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

  // Cofacteurs du Jacobien
  // Jacobian 伴随矩阵项
  cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
  cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
  cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
  cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);

  // Construction des dérivées des fonctions de forme
  // 形函数导数的构造（b[k][i]）
  b[0][0] = -cjxxi - cjxet - cjxze;
  b[0][1] =  cjxxi - cjxet - cjxze;
  b[0][2] =  cjxxi + cjxet - cjxze;
  b[0][3] = -cjxxi + cjxet - cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] = -cjyxi - cjyet - cjyze;
  b[1][1] =  cjyxi - cjyet - cjyze;
  b[1][2] =  cjyxi + cjyet - cjyze;
  b[1][3] = -cjyxi + cjyet - cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] = -cjzxi - cjzet - cjzze;
  b[2][1] =  cjzxi - cjzet - cjzze;
  b[2][2] =  cjzxi + cjzet - cjzze;
  b[2][3] = -cjzxi + cjzet - cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  // Volume = 8 * somme des produits de termes du Jacobien
  // 六面体单元体积公式（LULESH 传统推导）
  *volume = Real_t(8.0) * (fjxet * cjxet +
                           fjyet * cjyet +
                           fjzet * cjzet);
}

static inline void SumElemFaceNormal(
    Real_t* normalX0, Real_t* normalY0, Real_t* normalZ0,
    Real_t* normalX1, Real_t* normalY1, Real_t* normalZ1,
    Real_t* normalX2, Real_t* normalY2, Real_t* normalZ2,
    Real_t* normalX3, Real_t* normalY3, Real_t* normalZ3,
    const Real_t x0, const Real_t y0, const Real_t z0,
    const Real_t x1, const Real_t y1, const Real_t z1,
    const Real_t x2, const Real_t y2, const Real_t z2,
    const Real_t x3, const Real_t y3, const Real_t z3)
{
  // bisectX0 : vecteur bissecteur de la face dans la direction X
  // 面的对角线组合得到的“角平分向量”（几何构造的一部分）
  Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);

  // bisectY0 : idem pour la direction Y
  // 同理，面在 Y 方向上的角平分线量
  Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);

  // bisectZ0 : idem pour Z
  // 同理，面在 Z 方向上的角平分线量
  Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);

  // bisectX1 : deuxième vecteur bissecteur
  // 第二个角平分向量，用于构造面法向
  Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);

  // 同理
  Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
  Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);

  // areaX : composante X de la normale de la face
  // 面法向量 X 分量 = (bisectY0 × bisectZ1 - bisectZ0 × bisectY1) / 4
  Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);

  // areaY : composante Y de la normale
  // 面法向量 Y 分量
  Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);

  // areaZ : composante Z de la normale
  // 面法向量 Z 分量
  Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

  // Les quatre nœuds de la face reçoivent la même contribution normale
  // 面的 4 个节点都累加这同一个法向量贡献

  *normalX0 += areaX;
  *normalX1 += areaX;
  *normalX2 += areaX;
  *normalX3 += areaX;

  *normalY0 += areaY;
  *normalY1 += areaY;
  *normalY2 += areaY;
  *normalY3 += areaY;

  *normalZ0 += areaZ;
  *normalZ1 += areaZ;
  *normalZ2 += areaZ;
  *normalZ3 += areaZ;
}

KOKKOS_INLINE_FUNCTION
void CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8], Real_t pfz[8],
                         const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
  // Initialisation des normales nodales à zéro
  // 将八个节点的法向量初值全部设为 0
  for (Index_t i = 0; i < 8; ++i) {
    pfx[i] = Real_t(0.0);
    pfy[i] = Real_t(0.0);
    pfz[i] = Real_t(0.0);
  }

  // Face 0 : nœuds 0-1-2-3
  // 第 0 个面：节点 0-1-2-3
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                    &pfx[1], &pfy[1], &pfz[1],
                    &pfx[2], &pfy[2], &pfz[2],
                    &pfx[3], &pfy[3], &pfz[3],
                    x[0], y[0], z[0],
                    x[1], y[1], z[1],
                    x[2], y[2], z[2],
                    x[3], y[3], z[3]);

  // Face 1 : nœuds 0-4-5-1
  // 第 1 个面：节点 0-4-5-1
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                    &pfx[4], &pfy[4], &pfz[4],
                    &pfx[5], &pfy[5], &pfz[5],
                    &pfx[1], &pfy[1], &pfz[1],
                    x[0], y[0], z[0],
                    x[4], y[4], z[4],
                    x[5], y[5], z[5],
                    x[1], y[1], z[1]);

  // Face 2 : nœuds 1-5-6-2
  // 第 2 个面：节点 1-5-6-2
  SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                    &pfx[5], &pfy[5], &pfz[5],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[2], &pfy[2], &pfz[2],
                    x[1], y[1], z[1],
                    x[5], y[5], z[5],
                    x[6], y[6], z[6],
                    x[2], y[2], z[2]);

  // Face 3 : nœuds 2-6-7-3
  // 第 3 个面：节点 2-6-7-3
  SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[3], &pfy[3], &pfz[3],
                    x[2], y[2], z[2],
                    x[6], y[6], z[6],
                    x[7], y[7], z[7],
                    x[3], y[3], z[3]);

  // Face 4 : nœuds 3-7-4-0
  // 第 4 个面：节点 3-7-4-0
  SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[4], &pfy[4], &pfz[4],
                    &pfx[0], &pfy[0], &pfz[0],
                    x[3], y[3], z[3],
                    x[7], y[7], z[7],
                    x[4], y[4], z[4],
                    x[0], y[0], z[0]);

  // Face 5 : nœuds 4-7-6-5
  // 第 5 个面：节点 4-7-6-5
  SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[5], &pfy[5], &pfz[5],
                    x[4], y[4], z[4],
                    x[7], y[7], z[7],
                    x[6], y[6], z[6],
                    x[5], y[5], z[5]);
}

// Fonction : Somme des contraintes élémentaires vers les forces nodales
// 功能：将单元应力贡献累加到节点力向量
KOKKOS_INLINE_FUNCTION
void SumElemStressesToNodeForces(const Real_t B[][8],      // dérivées de fonctions de forme / 形函数导数
                                 const Real_t stress_xx,   // composante σxx / σxx 应力分量
                                 const Real_t stress_yy,   // composante σyy / σyy
                                 const Real_t stress_zz,   // composante σzz / σzz
                                 Real_t fx[],              // forces nodales x / x 方向节点力
                                 Real_t fy[],              // forces nodales y / y 方向节点力
                                 Real_t fz[])              // forces nodales z / z 方向节点力
{
  for (Index_t i = 0; i < 8; i++) {      // boucle sur 8 nœuds / 遍历全部 8 个节点
    fx[i] = -(stress_xx * B[0][i]);     // fx = -σxx * dN/dx
    fy[i] = -(stress_yy * B[1][i]);     // fy = -σyy * dN/dy
    fz[i] = -(stress_zz * B[2][i]);     // fz = -σzz * dN/dz
  }
}


// Fonction : intégration des contraintes pour chaque élément
// 功能：对所有单元整合应力，将它们转换为节点力
static inline
void IntegrateStressForElems(Domain &domain,     // domaine FEM / 有限元区域对象
                             Real_t *sigxx,      // σxx de chaque élément / 每个单元的 σxx
                             Real_t *sigyy,      // σyy de chaque élément
                             Real_t *sigzz,      // σzz de chaque élément
                             Real_t *determ,     // déterminant jacobien / 雅可比行列式
                             Index_t numElem,    // nombre d'éléments / 单元数量
                             Index_t numNode)    // nombre de nœuds / 节点数量
{
#if _OPENMP
  Index_t numthreads = omp_get_max_threads();   // nombre de threads OMP / OMP 线程数
#else
  Index_t numthreads = 1;                       // exécution séquentielle / 单线程执行
#endif

  Index_t numElem8 = numElem * 8;               // 8 nœuds par élément / 每单元 8 节点

  Real_t *fx_elem;
  Real_t *fy_elem;
  Real_t *fz_elem;

  // Vues Kokkos locales (taille 8)
  // Kokkos 局部 View（长度为 8，用于单线程情况）
  typedef Kokkos::View<Real_t *> view_real_t;
  view_real_t fx_local("A", 8);
  view_real_t fy_local("B", 8);
  view_real_t fz_local("C", 8);

  // Allocation CPU quand OpenMP utilise plusieurs threads
  // 若有多线程执行，则分配独立数组用于保存每单元的力结果
  if (numthreads > 1) {
    fx_elem = Allocate<Real_t>(numElem8);
    fy_elem = Allocate<Real_t>(numElem8);
    fz_elem = Allocate<Real_t>(numElem8);
  }

  // Boucle parallèle Kokkos sur les éléments
  // 使用 Kokkos 对所有单元进行并行处理
  Kokkos::parallel_for("IntegrateStressForElems A", numElem,
    KOKKOS_LAMBDA(const int k) {

      const Index_t *const elemToNode = domain.nodelist(k);  // connectivité élément->nœuds / 单元到节点映射

      Real_t B[3][8];          // dérivées des fonctions de forme / 形函数导数
      Real_t x_local[8];       // coordonnées x des nœuds locaux / 单元局部 x 坐标
      Real_t y_local[8];       // coordonnées y
      Real_t z_local[8];       // coordonnées z

      // Récupération des coordonnées nodales de l’élément
      // 将节点坐标拷贝到局部数组
      CollectDomainNodesToElemNodes(domain, elemToNode,
                                    x_local, y_local, z_local);

      // Calcul des dérivées des fonctions de forme + volume jacobien
      // 计算形函数导数与体积（雅可比 determinant）
      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                       B, &determ[k]);

      // Calcul des normales nodales de l’élément
      // 计算八个节点的法向量
      CalcElemNodeNormals(B[0], B[1], B[2],
                          x_local, y_local, z_local);

      // Cas multithread : écrire dans un grand tableau temporaire
      // 多线程情况：写入 fx_elem, fy_elem, fz_elem
      if (numthreads > 1) {

        SumElemStressesToNodeForces(B,
                                     sigxx[k], sigyy[k], sigzz[k],
                                     &fx_elem[k * 8],
                                     &fy_elem[k * 8],
                                     &fz_elem[k * 8]);
      }
      // Cas monothread : accumulation directe dans domain.fx/fy/fz
      // 单线程情况：直接累加到 Domain 中的节点力向量
      else {

        SumElemStressesToNodeForces(B,
                                    sigxx[k], sigyy[k], sigzz[k],
                                    &fx_local[8],
                                    &fy_local[8],
                                    &fz_local[8]);

        for (Index_t lnode = 0; lnode < 8; ++lnode) {
          Index_t gnode = elemToNode[lnode];  // ID global du nœud / 全局节点编号
          domain.fx(gnode) += fx_local[lnode]; // accumulation de forces / 力累加
          domain.fy(gnode) += fy_local[lnode];
          domain.fz(gnode) += fz_local[lnode];
        }
      }
  });

// Partie OpenMP/Kokkos : réduction finale des forces nodales
// 功能：在多线程模式下，将所有单元的力汇总到最终节点力向量
if (numthreads > 1) {

  // Boucle parallèle Kokkos sur tous les nœuds globaux
  // 用 Kokkos 并行遍历所有全局节点
  Kokkos::parallel_for("IntegrateStressForElems B", numNode,
    KOKKOS_LAMBDA(const int gnode) {

      Index_t count = domain.nodeElemCount(gnode);  
      // nombre d’occurrences de ce nœud dans les éléments / 此节点参与多少个单元

      Index_t *cornerList = domain.nodeElemCornerList(gnode);  
      // liste des coins d’élément associés / 关联该节点的单元角标列表

      Real_t fx_tmp = Real_t(0.0);  // accumulation locale fx / 本地 fx 累积
      Real_t fy_tmp = Real_t(0.0);  // accumulation locale fy / 本地 fy 累积
      Real_t fz_tmp = Real_t(0.0);  // accumulation locale fz / 本地 fz 累积

      // Boucle sur tous les coins d’éléments où le nœud apparaît
      // 遍历该节点对应的所有单元/角标
      for (Index_t i = 0; i < count; ++i) {

        Index_t ielem = cornerList[i];  
        // index global d’un coin d’élément / 单元角全局编号

        fx_tmp += fx_elem[ielem];  // somme des forces / 累加力
        fy_tmp += fy_elem[ielem];
        fz_tmp += fz_elem[ielem];
      }

      // Écriture du résultat dans le domaine
      // 将最终力写回 Domain
      domain.fx(gnode) = fx_tmp;
      domain.fy(gnode) = fy_tmp;
      domain.fz(gnode) = fz_tmp;
  });

  // Libération de la mémoire temporaire
  // 释放临时分配的数组
  Release(&fz_elem);
  Release(&fy_elem);
  Release(&fx_elem);
}
} 

// Fonction : dérivée volumique d’un élément hexaédrique
// 功能：计算六面体单元体积关于 x,y,z 的偏导数（用于 KDG/Hydro 算法）
static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t *dvdx, Real_t *dvdy, Real_t *dvdz)
{
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0);  
  // 1/12 用于几何对称化 / 体积公式需要的 1/12 系数

  // Formules géométriques directes pour la dérivée du volume
  // 几何公式：体积关于 x 的偏导数 dvdx
  *dvdx = (y1 + y2) * (z0 + z1)
        - (y0 + y1) * (z1 + z2)
        + (y0 + y4) * (z3 + z4)
        - (y3 + y4) * (z0 + z4)
        - (y2 + y5) * (z3 + z5)
        + (y3 + y5) * (z2 + z5);

  // dérivée dv/dy / 体积关于 y 的偏导数
  *dvdy = -( (x1 + x2) * (z0 + z1) )
          + (x0 + x1) * (z1 + z2)
          - (x0 + x4) * (z3 + z4)
          + (x3 + x4) * (z0 + z4)
          + (x2 + x5) * (z3 + z5)
          - (x3 + x5) * (z2 + z5);

  // dérivée dv/dz / 体积关于 z 的偏导数
  *dvdz = -( (y1 + y2) * (x0 + x1) )
          + (y0 + y1) * (x1 + x2)
          - (y0 + y4) * (x3 + x4)
          + (y3 + y4) * (x0 + x4)
          + (y2 + y5) * (x3 + x5)
          - (y3 + y5) * (x2 + x5);

  // Application du coefficient 1/12
  // 乘以 1/12 系数
  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}

// Fonction : calcule la dérivée volumique par nœud d’un élément hexaédrique.
// 功能：根据六面体拓扑，将 VoluDer 应用于 8 个节点的不同局部配置。
KOKKOS_INLINE_FUNCTION
void CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8], Real_t dvdz[8],
                              const Real_t x[8], const Real_t y[8],
                              const Real_t z[8]) {

  // 下面每一行都是对应 LULESH 原始几何的固定模式，不可改变拓扑顺序
  // Chaque appel VoluDer correspond à une face/rotation spécifique de l’élément.

  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
          y[1], y[2], y[3], y[4], y[5], y[7],
          z[1], z[2], z[3], z[4], z[5], z[7],
          &dvdx[0], &dvdy[0], &dvdz[0]);

  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
          y[0], y[1], y[2], y[7], y[4], y[6],
          z[0], z[1], z[2], z[7], z[4], z[6],
          &dvdx[3], &dvdy[3], &dvdz[3]);

  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
          y[3], y[0], y[1], y[6], y[7], y[5],
          z[3], z[0], z[1], z[6], z[7], z[5],
          &dvdx[2], &dvdy[2], &dvdz[2]);

  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
          y[2], y[3], y[0], y[5], y[6], y[4],
          z[2], z[3], z[0], z[5], z[6], z[4],
          &dvdx[1], &dvdy[1], &dvdz[1]);

  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
          y[7], y[6], y[5], y[0], y[3], y[1],
          z[7], z[6], z[5], z[0], z[3], z[1],
          &dvdx[4], &dvdy[4], &dvdz[4]);

  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
          y[4], y[7], y[6], y[1], y[0], y[2],
          z[4], z[7], z[6], z[1], z[0], z[2],
          &dvdx[5], &dvdy[5], &dvdz[5]);

  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
          y[5], y[4], y[7], y[2], y[1], y[3],
          z[5], z[4], z[7], z[2], z[1], z[3],
          &dvdx[6], &dvdy[6], &dvdz[6]);

  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
          y[6], y[5], y[4], y[3], y[2], y[0],
          z[6], z[5], z[4], z[3], z[2], z[0],
          &dvdx[7], &dvdy[7], &dvdz[7]);
}

// Fonction : calcul de la force hourglass FB (Flanagan-Belytschko)
// 功能：计算 FB 小时玻璃控制力（基于 4 模式 hourglass vectors）
KOKKOS_INLINE_FUNCTION
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                              Real_t hourgam[][4], Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz) {

  Real_t hxx[4];

  // Première passe : direction x
  // 第一步：沿 x 方向投影所有节点速度
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }

  // Force hourglass en x
  // 计算 x 方向 HG 力
  for (Index_t i = 0; i < 8; i++) {
    hgfx[i] = coefficient * (
      hourgam[i][0] * hxx[0] +
      hourgam[i][1] * hxx[1] +
      hourgam[i][2] * hxx[2] +
      hourgam[i][3] * hxx[3]);
  }

  // Deuxième passe : direction y
  // 第二步：沿 y 方向
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }

  // Force hourglass en y
  for (Index_t i = 0; i < 8; i++) {
    hgfy[i] = coefficient * (
      hourgam[i][0] * hxx[0] +
      hourgam[i][1] * hxx[1] +
      hourgam[i][2] * hxx[2] +
      hourgam[i][3] * hxx[3]);
  }

  // Troisième passe : direction z
  // 第三步：沿 z 方向
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }

  // Force hourglass en z
  for (Index_t i = 0; i < 8; i++) {
    hgfz[i] = coefficient * (
      hourgam[i][0] * hxx[0] +
      hourgam[i][1] * hxx[1] +
      hourgam[i][2] * hxx[2] +
      hourgam[i][3] * hxx[3]);
  }
}

// Fonction : force hourglass FB pour tous les éléments
// 功能：为所有单元计算 FB 小时玻璃控制力（Flanagan–Belytschko）
static inline
void CalcFBHourglassForceForElems(Domain &domain, Real_t *determ,
                                  Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                  Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                  Real_t hourg, Index_t numElem,
                                  Index_t numNode)
{
#if _OPENMP
  Index_t numthreads = omp_get_max_threads();  
  // Nombre de threads OpenMP / OpenMP 线程数
#else
  Index_t numthreads = 1;  
  // Mode séquentiel / 单线程模式
#endif

  Index_t numElem8 = numElem * 8;  
  // Nombre total de coefficients (8 par élément)

  Real_t *fx_elem;
  Real_t *fy_elem;
  Real_t *fz_elem;

  // Allocation seulement si on a du parallélisme externe
  // 若启用多线程，则分配 per-element 临时力数组
  if (numthreads > 1) {
    fx_elem = Allocate<Real_t>(numElem8);
    fy_elem = Allocate<Real_t>(numElem8);
    fz_elem = Allocate<Real_t>(numElem8);
  }

  // --------------------------
  // Matrice gamma 4x8 (modes hourglass)
  // 4 个小时玻璃模式，对应 8 节点权重
  // --------------------------
  Real_t gamma[4][8];

  gamma[0][0] = Real_t(1.);
  gamma[0][1] = Real_t(1.);
  gamma[0][2] = Real_t(-1.);
  gamma[0][3] = Real_t(-1.);
  gamma[0][4] = Real_t(-1.);
  gamma[0][5] = Real_t(-1.);
  gamma[0][6] = Real_t(1.);
  gamma[0][7] = Real_t(1.);

  gamma[1][0] = Real_t(1.);
  gamma[1][1] = Real_t(-1.);
  gamma[1][2] = Real_t(-1.);
  gamma[1][3] = Real_t(1.);
  gamma[1][4] = Real_t(-1.);
  gamma[1][5] = Real_t(1.);
  gamma[1][6] = Real_t(1.);
  gamma[1][7] = Real_t(-1.);

  gamma[2][0] = Real_t(1.);
  gamma[2][1] = Real_t(-1.);
  gamma[2][2] = Real_t(1.);
  gamma[2][3] = Real_t(-1.);
  gamma[2][4] = Real_t(1.);
  gamma[2][5] = Real_t(-1.);
  gamma[2][6] = Real_t(1.);
  gamma[2][7] = Real_t(-1.);

  gamma[3][0] = Real_t(-1.);
  gamma[3][1] = Real_t(1.);
  gamma[3][2] = Real_t(-1.);
  gamma[3][3] = Real_t(1.);
  gamma[3][4] = Real_t(1.);
  gamma[3][5] = Real_t(-1.);
  gamma[3][6] = Real_t(1.);
  gamma[3][7] = Real_t(-1.);
}

Kokkos::parallel_for("CalcFBHourglassForceForElems A", numElem,
  KOKKOS_LAMBDA(const int &i2) {

    // fx_local / fy_local / fz_local : tampon local (utilisé en mode séquentiel)
    // 本地缓存数组（仅用于单线程模式）
    Real_t *fx_local, *fy_local, *fz_local;

    // Forces hourglass pour 8 nœuds
    // 8 个节点的小时玻璃力
    Real_t hgfx[8], hgfy[8], hgfz[8];

    Real_t coefficient;  
    // coefficient : facteur d'échelle pour les forces hourglass
    // 小时玻璃力的缩放因子

    // hourgam : coefficients modifiés de la matrice hourglass (8 nœuds × 4 modes)
    // hourgam：修正后的小时玻璃矩阵（8 节点 × 4 模式）
    Real_t hourgam[8][4];

    // xd1, yd1, zd1 : vitesses nodales locales
    // 本地节点速度
    Real_t xd1[8], yd1[8], zd1[8];

    const Index_t *elemToNode = domain.nodelist(i2);
    // elemToNode : table de correspondance élément → nœuds
    // 单元 i2 对应的全局节点编号

    Index_t i3 = 8 * i2;
    // i3 : offset dans les tableaux x8n[], y8n[], z8n[] pour cet élément
    // 当前单元在展开数组中的偏移量

    Real_t volinv = Real_t(1.0) / determ[i2];
    // volinv : inverse du volume du tétraèdre / 当前单元体积的倒数

    Real_t ss1, mass1, volume13;
    // ss1 : sound speed / 声速
    // mass1 : masse de l’élément / 单元质量
    // volume13 : racine cubique du volume / 体积的立方根

    // 1) Construction des coefficients hourglass corrigés (hourgam)
    // 计算修正后的小时玻璃模式 hourgam
    // Chaque mode hourglass (4 modes) doit être orthogonalisé via dvdx/dvdy/dvdz
    // 每一个小时玻璃模式（4 模式）都必须通过 dv* 向量进行正交修正
    for (Index_t i1 = 0; i1 < 4; ++i1) {

      // hourmodx/y/z : projection du mode hourglass sur les coordonnées x,y,z
      // hourmodx/y/z：模式与节点几何的投影
      Real_t hourmodx =
        x8n[i3]     * gamma[i1][0] +
        x8n[i3 + 1] * gamma[i1][1] +
        x8n[i3 + 2] * gamma[i1][2] +
        x8n[i3 + 3] * gamma[i1][3] +
        x8n[i3 + 4] * gamma[i1][4] +
        x8n[i3 + 5] * gamma[i1][5] +
        x8n[i3 + 6] * gamma[i1][6] +
        x8n[i3 + 7] * gamma[i1][7];

      Real_t hourmody =
        y8n[i3]     * gamma[i1][0] +
        y8n[i3 + 1] * gamma[i1][1] +
        y8n[i3 + 2] * gamma[i1][2] +
        y8n[i3 + 3] * gamma[i1][3] +
        y8n[i3 + 4] * gamma[i1][4] +
        y8n[i3 + 5] * gamma[i1][5] +
        y8n[i3 + 6] * gamma[i1][6] +
        y8n[i3 + 7] * gamma[i1][7];

      Real_t hourmodz =
        z8n[i3]     * gamma[i1][0] +
        z8n[i3 + 1] * gamma[i1][1] +
        z8n[i3 + 2] * gamma[i1][2] +
        z8n[i3 + 3] * gamma[i1][3] +
        z8n[i3 + 4] * gamma[i1][4] +
        z8n[i3 + 5] * gamma[i1][5] +
        z8n[i3 + 6] * gamma[i1][6] +
        z8n[i3 + 7] * gamma[i1][7];

      // hourgam[?][i1] = gamma - volinv * (dvdx*dX + dvdy*dY + dvdz*dZ)
      // 修正 hourglass 模式，使其不改变单元体积（体积正交化）
      hourgam[0][i1] =
        gamma[i1][0] -
        volinv * (dvdx[i3]     * hourmodx +
                  dvdy[i3]     * hourmody +
                  dvdz[i3]     * hourmodz);

      hourgam[1][i1] =
        gamma[i1][1] -
        volinv * (dvdx[i3 + 1] * hourmodx +
                  dvdy[i3 + 1] * hourmody +
                  dvdz[i3 + 1] * hourmodz);

      hourgam[2][i1] =
        gamma[i1][2] -
        volinv * (dvdx[i3 + 2] * hourmodx +
                  dvdy[i3 + 2] * hourmody +
                  dvdz[i3 + 2] * hourmodz);

      hourgam[3][i1] =
        gamma[i1][3] -
        volinv * (dvdx[i3 + 3] * hourmodx +
                  dvdy[i3 + 3] * hourmody +
                  dvdz[i3 + 3] * hourmodz);

      hourgam[4][i1] =
        gamma[i1][4] -
        volinv * (dvdx[i3 + 4] * hourmodx +
                  dvdy[i3 + 4] * hourmody +
                  dvdz[i3 + 4] * hourmodz);

      hourgam[5][i1] =
        gamma[i1][5] -
        volinv * (dvdx[i3 + 5] * hourmodx +
                  dvdy[i3 + 5] * hourmody +
                  dvdz[i3 + 5] * hourmodz);

      hourgam[6][i1] =
        gamma[i1][6] -
        volinv * (dvdx[i3 + 6] * hourmodx +
                  dvdy[i3 + 6] * hourmody +
                  dvdz[i3 + 6] * hourmodz);

      hourgam[7][i1] =
        gamma[i1][7] -
        volinv * (dvdx[i3 + 7] * hourmodx +
                  dvdy[i3 + 7] * hourmody +
                  dvdz[i3 + 7] * hourmodz);
    }

    // 2) Lecture des constantes physiques : ss (sound speed), mass, volume
    // 读取单元物理常数：声速、单元质量、体积立方根
    ss1 = domain.ss(i2);
    mass1 = domain.elemMass(i2);
    volume13 = CBRT(determ[i2]);

    // 3) Chargement des vitesses nodales pour les 8 nœuds du hexaèdre
    // 读取 8 个节点的速度
    Index_t n0si2 = elemToNode[0];
    Index_t n1si2 = elemToNode[1];
    Index_t n2si2 = elemToNode[2];
    Index_t n3si2 = elemToNode[3];
    Index_t n4si2 = elemToNode[4];
    Index_t n5si2 = elemToNode[5];
    Index_t n6si2 = elemToNode[6];
    Index_t n7si2 = elemToNode[7];

    xd1[0] = domain.xd(n0si2);
    xd1[1] = domain.xd(n1si2);
    xd1[2] = domain.xd(n2si2);
    xd1[3] = domain.xd(n3si2);
    xd1[4] = domain.xd(n4si2);
    xd1[5] = domain.xd(n5si2);
    xd1[6] = domain.xd(n6si2);
    xd1[7] = domain.xd(n7si2);

    yd1[0] = domain.yd(n0si2);
    yd1[1] = domain.yd(n1si2);
    yd1[2] = domain.yd(n2si2);
    yd1[3] = domain.yd(n3si2);
    yd1[4] = domain.yd(n4si2);
    yd1[5] = domain.yd(n5si2);
    yd1[6] = domain.yd(n6si2);
    yd1[7] = domain.yd(n7si2);

    zd1[0] = domain.zd(n0si2);
    zd1[1] = domain.zd(n1si2);
    zd1[2] = domain.zd(n2si2);
    zd1[3] = domain.zd(n3si2);
    zd1[4] = domain.zd(n4si2);
    zd1[5] = domain.zd(n5si2);
    zd1[6] = domain.zd(n6si2);
    zd1[7] = domain.zd(n7si2);

        // coefficient : coefficient global des forces hourglass
    // 小时玻璃力的全局系数
    coefficient = -hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    // Calcul des forces hourglass pour cet élément
    // 计算该单元的小时玻璃力
    CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient,
                             hgfx, hgfy, hgfz);

    // Si on utilise plusieurs threads OpenMP : stockage temporaire dans fx_elem/fy_elem/fz_elem
    // 如果使用多线程模式：将结果写入 fx_elem/fy_elem/fz_elem（延后归并）
    if (numthreads > 1) {

      // fx_local : pointeur vers la zone locale de l’élément i3 dans fx_elem
      // fx_local：指向当前单元 i3 的 fx_elem 局部区域
      fx_local = &fx_elem[i3];
      fx_local[0] = hgfx[0];
      fx_local[1] = hgfx[1];
      fx_local[2] = hgfx[2];
      fx_local[3] = hgfx[3];
      fx_local[4] = hgfx[4];
      fx_local[5] = hgfx[5];
      fx_local[6] = hgfx[6];
      fx_local[7] = hgfx[7];

      // fy_local : idem pour fy_elem
      // fy_local 同理，写 fy_elem
      fy_local = &fy_elem[i3];
      fy_local[0] = hgfy[0];
      fy_local[1] = hgfy[1];
      fy_local[2] = hgfy[2];
      fy_local[3] = hgfy[3];
      fy_local[4] = hgfy[4];
      fy_local[5] = hgfy[5];
      fy_local[6] = hgfy[6];
      fy_local[7] = hgfy[7];

      // fz_local : idem pour fz_elem
      // fz_local 同理，写 fz_elem
      fz_local = &fz_elem[i3];
      fz_local[0] = hgfz[0];
      fz_local[1] = hgfz[1];
      fz_local[2] = hgfz[2];
      fz_local[3] = hgfz[3];
      fz_local[4] = hgfz[4];
      fz_local[5] = hgfz[5];
      fz_local[6] = hgfz[6];
      fz_local[7] = hgfz[7];

    } else {

      // Mode séquentiel : on applique directement la force sur les nœuds du domaine
      // 单线程模式：直接将小时玻璃力加到域节点上

      domain.fx(n0si2) += hgfx[0];
      domain.fy(n0si2) += hgfy[0];
      domain.fz(n0si2) += hgfz[0];

      domain.fx(n1si2) += hgfx[1];
      domain.fy(n1si2) += hgfy[1];
      domain.fz(n1si2) += hgfz[1];

      domain.fx(n2si2) += hgfx[2];
      domain.fy(n2si2) += hgfy[2];
      domain.fz(n2si2) += hgfz[2];

      domain.fx(n3si2) += hgfx[3];
      domain.fy(n3si2) += hgfy[3];
      domain.fz(n3si2) += hgfz[3];

      domain.fx(n4si2) += hgfx[4];
      domain.fy(n4si2) += hgfy[4];
      domain.fz(n4si2) += hgfz[4];

      domain.fx(n5si2) += hgfx[5];
      domain.fy(n5si2) += hgfy[5];
      domain.fz(n5si2) += hgfz[5];

      domain.fx(n6si2) += hgfx[6];
      domain.fy(n6si2) += hgfy[6];
      domain.fz(n6si2) += hgfz[6];

      domain.fx(n7si2) += hgfx[7];
      domain.fy(n7si2) += hgfy[7];
      domain.fz(n7si2) += hgfz[7];
    }
  });

// Si plusieurs threads OpenMP sont utilisés : réduction parallèle des forces sur chaque nœud
// 如果启用多线程 OpenMP：对每个节点的力进行并行归并
if (numthreads > 1) {

    Kokkos::parallel_for("CalcFBHourglassForceForElems B", numNode,
                         KOKKOS_LAMBDA(const int gnode) {

        // count : nombre d'éléments connectés à ce nœud
        // count：该节点关联的单元数
        Index_t count = domain.nodeElemCount(gnode);

        // cornerList : liste des indices (8*k + lnode) correspondant aux coins associés
        // cornerList：当前节点对应的 corner 索引列表（即元素局部节点号展平后的索引）
        Index_t* cornerList = domain.nodeElemCornerList(gnode);

        // fx_tmp, fy_tmp, fz_tmp : accumulation temporaire des forces
        // fx_tmp, fy_tmp, fz_tmp：用于累加来自所有单元的节点力
        Real_t fx_tmp = Real_t(0.0);
        Real_t fy_tmp = Real_t(0.0);
        Real_t fz_tmp = Real_t(0.0);

        // Accumulation des contributions
        // 累加每个关联 corner 的贡献
        for (Index_t i = 0; i < count; ++i) {
            Index_t ielem = cornerList[i];
            fx_tmp += fx_elem[ielem];
            fy_tmp += fy_elem[ielem];
            fz_tmp += fz_elem[ielem];
        }

        // Mise à jour des forces nodales globales
        // 更新域中该节点的全局力
        domain.fx(gnode) += fx_tmp;
        domain.fy(gnode) += fy_tmp;
        domain.fz(gnode) += fz_tmp;
    });

    // Libération des tableaux temporaires
    // 释放线程模式下使用的临时数组
    Release(&fz_elem);
    Release(&fy_elem);
    Release(&fx_elem);
}
}

// Fonction de contrôle des hourglass pour tous les éléments
// 对所有单元执行 Hourglass 控制力计算
static inline void CalcHourglassControlForElems(Domain& domain,
                                                Real_t determ[],
                                                Real_t hgcoef) {

    // numElem : nombre total d’éléments
    // numElem：单元总数
    Index_t numElem = domain.numElem();

    // numElem8 : nombre total de valeurs (8 par élément)
    // numElem8：总长度（每个单元 8 个节点）
    Index_t numElem8 = numElem * 8;

    // Allocation des dérivées du volume par nœud
    // 分配体积导数数组
    Real_t* dvdx = Allocate<Real_t>(numElem8);
    Real_t* dvdy = Allocate<Real_t>(numElem8);
    Real_t* dvdz = Allocate<Real_t>(numElem8);

    // Coordonnées des 8 nœuds pour chaque élément (aplaties)
    // 存储每个单元的 8 个节点坐标（展平格式）
    Real_t* x8n = Allocate<Real_t>(numElem8);
    Real_t* y8n = Allocate<Real_t>(numElem8);
    Real_t* z8n = Allocate<Real_t>(numElem8);

    // Boucle parallèle sur les éléments
    // 对所有单元执行并行循环
    Kokkos::parallel_for(numElem, KOKKOS_LAMBDA(const int i) {

        // x1, y1, z1 : positions nodales locales (taille 8)
        // x1, y1, z1：该单元 8 个节点的坐标
        Real_t x1[8], y1[8], z1[8];

        // pfx, pfy, pfz : dérivées du volume
        // pfx, pfy, pfz：体积导数
        Real_t pfx[8], pfy[8], pfz[8];

        // elemToNode : mapping vers les 8 nœuds globaux
        // elemToNode：全局 8 节点 ID 映射
        Index_t* elemToNode = domain.nodelist(i);

        // Récupère les coordonnées nodales
        // 收集单元的节点坐标
        CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

        // Calcule les dérivées du volume dV/dx, dV/dy, dV/dz
        // 计算体积导数
        CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

        // Stocke les valeurs (aplaties dans dvdx/dvdy/dvdz)
        // 将结果写入展平数组
        for (Index_t ii = 0; ii < 8; ++ii) {
            Index_t jj = 8 * i + ii;

            dvdx[jj] = pfx[ii];
            dvdy[jj] = pfy[ii];
            dvdz[jj] = pfz[ii];
            x8n[jj]  = x1[ii];
            y8n[jj]  = y1[ii];
            z8n[jj]  = z1[ii];
        }

        // determ[i] = volume de référence * volume relatif actuel
        // determ[i] = 初始体积 * 当前相对体积
        determ[i] = domain.volo(i) * domain.v(i);

        // Si le volume devient négatif : erreur fatale
        // 如果体积为负，触发致命错误
        if (domain.v(i) <= Real_t(0.0)) {
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
            exit(VolumeError);
#endif
        }
    });

    // Si coef hourglass > 0 : appliquer les forces hourglass
    // 若开启 HG 控制：计算 Hourglass 力
    if (hgcoef > Real_t(0.)) {
        CalcFBHourglassForceForElems(domain, determ,
                                     x8n, y8n, z8n,
                                     dvdx, dvdy, dvdz,
                                     hgcoef, numElem,
                                     domain.numNode());
    }

    // Libération des tableaux alloués
    // 释放进行 HG 计算时分配的所有临时数组
    Release(&z8n);
    Release(&y8n);
    Release(&x8n);
    Release(&dvdz);
    Release(&dvdy);
    Release(&dvdx);

    return;
}

// Fonction : CalcVolumeForceForElems
// 作用：计算单元体积力（压力 + 黏性 + hourglass）并累积到节点力
static inline void CalcVolumeForceForElems(Domain &domain) {

  // numElem : nombre total d'éléments
  // numElem：单元总数
  Index_t numElem = domain.numElem();

  if (numElem != 0) {

    // hgcoef : coefficient hourglass global
    // hgcoef：Hourglass 总系数
    Real_t hgcoef = domain.hgcoef();

    // Allocation des tableaux de contraintes et de déterminants
    // 分配应力与体积行列式数组
    Real_t* sigxx = Allocate<Real_t>(numElem);
    Real_t* sigyy = Allocate<Real_t>(numElem);
    Real_t* sigzz = Allocate<Real_t>(numElem);
    Real_t* determ = Allocate<Real_t>(numElem);

    // Initialise sigxx, sigyy, sigzz = -(p + q)
    // 初始化应力项 sigxx = sigyy = sigzz = - (p + q)
    InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

    // Calcule les forces de volume par intégration des contraintes
    // 通过积分应力计算体积力
    IntegrateStressForElems(domain, sigxx, sigyy, sigzz,
                            determ, numElem, domain.numNode());

    // Vérification du signe du déterminant (volume)
    // 检查体积是否为正，否则报错
    Kokkos::parallel_for(numElem, KOKKOS_LAMBDA(const int k) {
      if (determ[k] <= Real_t(0.0)) {
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
        exit(VolumeError);
#endif
      }
    });

    // Calcul des forces hourglass
    // 计算 Hourglass 力
    CalcHourglassControlForElems(domain, determ, hgcoef);

    // Libération
    // 释放内存
    Release(&determ);
    Release(&sigzz);
    Release(&sigyy);
    Release(&sigxx);
  }
}

// Fonction : CalcForceForNodes
// 作用：清零节点力 → 计算体积力 → MPI 边界通信
static inline void CalcForceForNodes(Domain &domain) {

  // numNode : nombre total de nœuds
  // numNode：节点总数
  Index_t numNode = domain.numNode();

#if USE_MPI
  // Réception des valeurs fantômes avant le calcul local
  // 在本地计算前接收 Ghost 值
  CommRecv(domain, MSG_COMM_SBN, 3,
           domain.sizeX() + 1,
           domain.sizeY() + 1,
           domain.sizeZ() + 1,
           true, false);
#endif

  // Mise à zéro des forces nodales
  // 将所有节点力设为 0
  Kokkos::parallel_for("CalcForceForNodes", numNode,
                       KOKKOS_LAMBDA(const int i) {
    domain.fx(i) = Real_t(0.0);
    domain.fy(i) = Real_t(0.0);
    domain.fz(i) = Real_t(0.0);
  });

  // Calcul complet des forces (pression, viscosité, hourglass)
  // 计算完整的节点力（压力 + 黏性 + hourglass）
  CalcVolumeForceForElems(domain);

#if USE_MPI

  // fieldData : pointeurs vers les champs fx, fy, fz du domaine
  // fieldData：指向 fx, fy, fz 成员函数的指针（用于通信）
  Domain_member fieldData[3];
  fieldData[0] = &Domain::fx;
  fieldData[1] = &Domain::fy;
  fieldData[2] = &Domain::fz;

  // Envoi des valeurs fantômes
  // 发送 Ghost 区节点力
  CommSend(domain, MSG_COMM_SBN, 3,
           fieldData,
           domain.sizeX() + 1,
           domain.sizeY() + 1,
           domain.sizeZ() + 1,
           true, false);

  // Synchronisation des buffers nodaux
  // 完成边界节点力的同步
  CommSBN(domain, 3, fieldData);

#endif
}

// Fonction : CalcAccelerationForNodes
// 通过 fx/mass 等式计算三个方向的加速度
static inline void CalcAccelerationForNodes(Domain &domain, Index_t numNode) {

  // 并行遍历所有节点，直接写入 domain.xdd / ydd / zdd
  Kokkos::parallel_for("CalcAccelerationForNodes", numNode,
                       KOKKOS_LAMBDA(const int i) {
    // 加速度 = 力 / 质量
    domain.xdd(i) = domain.fx(i) / domain.nodalMass(i);
    domain.ydd(i) = domain.fy(i) / domain.nodalMass(i);
    domain.zdd(i) = domain.fz(i) / domain.nodalMass(i);
  });
}


// Fonction : ApplyAccelerationBoundaryConditionsForNodes
// 对称边界的节点加速度必须设为 0
static inline void ApplyAccelerationBoundaryConditionsForNodes(Domain &domain) {

  Index_t size = domain.sizeX();
  Index_t numNodeBC = (size + 1) * (size + 1); 
  // 一个面 (X=0, Y=0 或 Z=0 面) 上的节点数量

  // 对称 X 边界：所有在 X=0 面上的节点，其 x 方向加速度 = 0
  if (!domain.symmXempty() != 0) {
    Kokkos::parallel_for("ApplyAccelerationBoundaryConditionsForNodes A",
                         numNodeBC, KOKKOS_LAMBDA(const int i) {
      domain.xdd(domain.symmX(i)) = Real_t(0.0);
    });
  }

  // 对称边界对应的节点加速度被强制为 0
  if (!domain.symmYempty() != 0) {
    Kokkos::parallel_for("ApplyAccelerationBoundaryConditionsForNodes B",
                         numNodeBC, KOKKOS_LAMBDA(const int i) {
      domain.ydd(domain.symmY(i)) = Real_t(0.0);
    });
  }

  // 如果 Z 对称边界非空，则置零所有 zdd
  if (!domain.symmZempty() != 0) {
    Kokkos::parallel_for("ApplyAccelerationBoundaryConditionsForNodes C",
                         numNodeBC, KOKKOS_LAMBDA(const int i) {
      domain.zdd(domain.symmZ(i)) = Real_t(0.0);
    });
  }
}


// Fonction : CalcVelocityForNodes
// 速度更新公式 xd += xdd * dt；并应用速度截断 u_cut
static inline void CalcVelocityForNodes(Domain &domain, const Real_t dt,
                                        const Real_t u_cut, Index_t numNode) {

  // 每个节点独立更新速度，因此非常适合 Kokkos 并行
  Kokkos::parallel_for("CalcVelocityForNodes", numNode,
                       KOKKOS_LAMBDA(const int i) {

    Real_t xdtmp, ydtmp, zdtmp;

    // vx(t+dt) = vx(t) + ax * dt
    xdtmp = domain.xd(i) + domain.xdd(i) * dt;

    // 如果速度很小（低于 u_cut），认为其为 0，以抑制数值噪声
    if (FABS(xdtmp) < u_cut)
      xdtmp = Real_t(0.0);
    domain.xd(i) = xdtmp;

    // y 方向速度更新
    ydtmp = domain.yd(i) + domain.ydd(i) * dt;
    if (FABS(ydtmp) < u_cut)
      ydtmp = Real_t(0.0);
    domain.yd(i) = ydtmp;

    // z 方向速度更新
    zdtmp = domain.zd(i) + domain.zdd(i) * dt;
    if (FABS(zdtmp) < u_cut)
      zdtmp = Real_t(0.0);
    domain.zd(i) = zdtmp;
  });
}

// Fonction : CalcPositionForNodes
// x(t+dt) = x(t) + v(t)*dt，三个方向独立更新
static inline void CalcPositionForNodes(Domain &domain, const Real_t dt,
                                        Index_t numNode) {

  // Kokkos 并行地更新所有节点的位置
  Kokkos::parallel_for("CalcPositionForNodes", numNode,
                       KOKKOS_LAMBDA(const int i) {
    domain.x(i) += domain.xd(i) * dt;   // 法语：更新 X 坐标
    domain.y(i) += domain.yd(i) * dt;   // 中文：更新 Y 坐标
    domain.z(i) += domain.zd(i) * dt;   // 中文：更新 Z 坐标
  });
}


// Fonction : LagrangeNodal 
// 这是显式 Lagrange 时间推进中负责处理节点物理量的部分
static inline void LagrangeNodal(Domain &domain) {

#ifdef SEDOV_SYNC_POS_VEL_EARLY
  // MPI 情况下，如果启用了“早期同步”，需要准备要同步的字段
  Domain_member fieldData[6];
#endif

  const Real_t delt = domain.deltatime();  // 法语：本步长度 Δt
  Real_t u_cut = domain.u_cut();           // 法语：速度截断阈值

  // 计算所有节点的力（包括应力、人工粘度、体积力等）
  CalcForceForNodes(domain);

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  // 接收来自相邻 MPI 进程的 x,y,z,xd,yd,zd（同步）
  CommRecv(domain, MSG_SYNC_POS_VEL, 6,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
           false, false);
#endif
#endif

  // 更新 xdd, ydd, zdd
  CalcAccelerationForNodes(domain, domain.numNode());

  // 施加加速度边界条件，如对称边界（某些方向加速度必须为 0）
  ApplyAccelerationBoundaryConditionsForNodes(domain);

  // xd += xdd*dt，并使用 u_cut 截断速度
  CalcVelocityForNodes(domain, delt, u_cut, domain.numNode());

  // 根据新速度推进节点位置
  CalcPositionForNodes(domain, delt, domain.numNode());

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  // 设置要发送给其他 MPI 进程的字段
  fieldData[0] = &Domain::x;
  fieldData[1] = &Domain::y;
  fieldData[2] = &Domain::z;
  fieldData[3] = &Domain::xd;
  fieldData[4] = &Domain::yd;
  fieldData[5] = &Domain::zd;

  // 发送 x,y,z,xd,yd,zd 到邻域进程
  CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
           false, false);

  // 执行实际的同步
  CommSyncPosVel(domain);
#endif
#endif

  return;
}

// Fonction : CalcElemVolume
// 用六面体体积分解公式计算体积，是 LULESH 的标准实现。
KOKKOS_INLINE_FUNCTION
Real_t CalcElemVolume(
    const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
    const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
    const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
    const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
    const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
    const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7)
{
  // LULESH 体积计算最终需要乘 1/12
  Real_t twelveth = Real_t(1.0) / Real_t(12.0);

  // 以下计算各种边向量（节点间的差）
  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

  // TRIPLE_PRODUCT(a,b,c) = a · (b × c)
#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
  ((x1) * ((y2) * (z3) - (z2) * (y3)) +                     \
   (x2) * ((z1) * (y3) - (y1) * (z3)) +                     \
   (x3) * ((y1) * (z2) - (z1) * (y2)))

  // 体积的核心公式（来自六面体分解成三个平行六面体）
  Real_t volume =
      TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
                     dy31 + dy72, dy63, dy20,
                     dz31 + dz72, dz63, dz20)
    +
      TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
                     dy43 + dy57, dy64, dy70,
                     dz43 + dz57, dz64, dz70)
    +
      TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
                     dy14 + dy25, dy61, dy50,
                     dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  // 最终体积乘以 1/12
  volume *= twelveth;

  return volume;
}

// 调用坐标数组版本，转发到 24 参数版本
// Fonction d’appoint qui redirige vers la version à 24 arguments.
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8]) {
  // 按节点顺序展开，并调用 CalcElemVolume(...) 主版本
  // Déroule les coordonnées nodales et appelle la version principale.
  return CalcElemVolume(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                        z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

// 计算四边形面的“面积度量”，用于单元特征长度计算
// Calcule une mesure de surface d’une face quadrilatérale, utilisée pour la longueur caractéristique.
static inline Real_t AreaFace(const Real_t x0, const Real_t x1, const Real_t x2,
                              const Real_t x3, const Real_t y0, const Real_t y1,
                              const Real_t y2, const Real_t y3, const Real_t z0,
                              const Real_t z1, const Real_t z2,
                              const Real_t z3) {
  // fx = (x2 - x0) - (x3 - x1)
  Real_t fx = (x2 - x0) - (x3 - x1);

  // fy 与 fz 同样是“反对角向量差”
  // fy et fz représentent aussi une différence entre diagonales.
  Real_t fy = (y2 - y0) - (y3 - y1);
  Real_t fz = (z2 - z0) - (z3 - z1);

  // gx = (x2 - x0) + (x3 - x1)
  // gx = (x2 - x0) + (x3 - x1)
  Real_t gx = (x2 - x0) + (x3 - x1);

  // gy、gz 为“对角和”
  // gy et gz sont les sommes diagonales.
  Real_t gy = (y2 - y0) + (y3 - y1);
  Real_t gz = (z2 - z0) + (z3 - z1);

  // 面积度量 = |f|² |g|² − (f·g)²
  // mesure de surface = |f|² |g|² − (f·g)²
  Real_t area =
      (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);

  // 返回面积度量（注意，不是实际面积）
  // Retourne la mesure (ce n’est pas l’aire réelle).
  return area;
}

// 计算单元的特征长度（基于最大面面积度量）
// Calcule la longueur caractéristique de l’élément (basée sur la plus grande mesure de face).
static inline Real_t
CalcElemCharacteristicLength(const Real_t x[8], const Real_t y[8],
                             const Real_t z[8], const Real_t volume) {
  // a 为当前面面积度量；charLength 保存最大值
  // a est la mesure de face ; charLength conserve la valeur maximale.
  Real_t a, charLength = Real_t(0.0);

  // Face 0–1–2–3
  // 检查第 1 个面
  // Première face.
  a = AreaFace(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3],
               z[0], z[1], z[2], z[3]);
  charLength = std::max(a, charLength);

  a = AreaFace(x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7],
               z[4], z[5], z[6], z[7]);
  charLength = std::max(a, charLength);

  a = AreaFace(x[0], x[1], x[5], x[4], y[0], y[1], y[5], y[4],
               z[0], z[1], z[5], z[4]);
  charLength = std::max(a, charLength);

  a = AreaFace(x[1], x[2], x[6], x[5], y[1], y[2], y[6], y[5],
               z[1], z[2], z[6], z[5]);
  charLength = std::max(a, charLength);

  a = AreaFace(x[2], x[3], x[7], x[6], y[2], y[3], y[7], y[6],
               z[2], z[3], z[7], z[6]);
  charLength = std::max(a, charLength);

  a = AreaFace(x[3], x[0], x[4], x[7], y[3], y[0], y[4], y[7],
               z[3], z[0], z[4], z[7]);
  charLength = std::max(a, charLength);

  // 最终特征长度 = 4 * 体积 / sqrt(最大面积度量)
  // Longueur caractéristique finale = 4 * volume / sqrt(max(face)).
  charLength = Real_t(4.0) * volume / SQRT(charLength);

  return charLength;
}

// 计算单元速度梯度（∂vx/∂x, ∂vy/∂y 等）
// Calcule le gradient de vitesse dans l’élément.
static inline void
CalcElemVelocityGradient(const Real_t *const xvel, const Real_t *const yvel,
                         const Real_t *const zvel, const Real_t b[][8],
                         const Real_t detJ, Real_t *const d) {

  // Jacobian 行列式的倒数
  // Inverse du déterminant du jacobien.
  const Real_t inv_detJ = Real_t(1.0) / detJ;

  // 临时变量，用于存储中间梯度项
  // Variables temporaires pour les gradients intermédiaires.
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;

  // b 矩阵的三个分量 (形函数导数)
  // Composantes du tableau b (dérivées des fonctions de forme).
  const Real_t *const pfx = b[0];
  const Real_t *const pfy = b[1];
  const Real_t *const pfz = b[2];

  // d[0] = ∂vx/∂x
  // d[0] 表示速度分量 x 对 x 的偏导
  // d[0] représente ∂vx/∂x.
  d[0] = inv_detJ * (pfx[0] * (xvel[0] - xvel[6]) +
                     pfx[1] * (xvel[1] - xvel[7]) +
                     pfx[2] * (xvel[2] - xvel[4]) +
                     pfx[3] * (xvel[3] - xvel[5]));

  d[1] = inv_detJ * (pfy[0] * (yvel[0] - yvel[6]) +
                     pfy[1] * (yvel[1] - yvel[7]) +
                     pfy[2] * (yvel[2] - yvel[4]) +
                     pfy[3] * (yvel[3] - yvel[5]));

  d[2] = inv_detJ * (pfz[0] * (zvel[0] - zvel[6]) +
                     pfz[1] * (zvel[1] - zvel[7]) +
                     pfz[2] * (zvel[2] - zvel[4]) +
                     pfz[3] * (zvel[3] - zvel[5]));

  // 混合导数部分 
  // 以下为 y 对 x、x 对 y、z 对 x 等的偏导
  // Les dérivées croisées ci-dessous.

  dyddx = inv_detJ * (pfx[0] * (yvel[0] - yvel[6]) +
                      pfx[1] * (yvel[1] - yvel[7]) +
                      pfx[2] * (yvel[2] - yvel[4]) +
                      pfx[3] * (yvel[3] - yvel[5]));

  dxddy = inv_detJ * (pfy[0] * (xvel[0] - xvel[6]) +
                      pfy[1] * (xvel[1] - xvel[7]) +
                      pfy[2] * (xvel[2] - xvel[4]) +
                      pfy[3] * (xvel[3] - xvel[5]));

  dzddx = inv_detJ * (pfx[0] * (zvel[0] - zvel[6]) +
                      pfx[1] * (zvel[1] - zvel[7]) +
                      pfx[2] * (zvel[2] - zvel[4]) +
                      pfx[3] * (zvel[3] - zvel[5]));

  dxddz = inv_detJ * (pfz[0] * (xvel[0] - xvel[6]) +
                      pfz[1] * (xvel[1] - xvel[7]) +
                      pfz[2] * (xvel[2] - xvel[4]) +
                      pfz[3] * (xvel[3] - xvel[5]));

  dzddy = inv_detJ * (pfy[0] * (zvel[0] - zvel[6]) +
                      pfy[1] * (zvel[1] - zvel[7]) +
                      pfy[2] * (zvel[2] - zvel[4]) +
                      pfy[3] * (zvel[3] - zvel[5]));

  dyddz = inv_detJ * (pfz[0] * (yvel[0] - yvel[6]) +
                      pfz[1] * (yvel[1] - yvel[7]) +
                      pfz[2] * (yvel[2] - yvel[4]) +
                      pfz[3] * (yvel[3] - yvel[5]));

  // 构造对称部分
  // d[5], d[4], d[3] 是三维速度梯度张量的对称项
  // d[5], d[4], d[3] sont les termes symétriques du tenseur de gradient.
  d[5] = Real_t(.5) * (dxddy + dyddx);  // ∂vx/∂y + ∂vy/∂x
  d[4] = Real_t(.5) * (dxddz + dzddx);  // ∂vx/∂z + ∂vz/∂x
  d[3] = Real_t(.5) * (dzddy + dyddz);  // ∂vy/∂z + ∂vz/∂y
}

// 计算单元运动学量：体积变化、形函数导数、速度梯度
// Calcule les grandeurs cinématiques des éléments : variation de volume, dérivées des fonctions de forme et gradient de vitesse.
void CalcKinematicsForElems(Domain &domain, Real_t deltaTime, Index_t numElem) {

  // 对每个单元执行运动学更新
  // Exécute la mise à jour cinématique pour chaque élément.
  Kokkos::parallel_for("CalcKinematicsForElems", numElem,
                       KOKKOS_LAMBDA(const int k) {

    // 局部形函数导数矩阵 B（3×8）  
    // Matrice locale des dérivées de forme B (3×8).
    Real_t B[3][8];

    // 局部速度梯度分量 D（长度 6）
    // Composantes locales du gradient de vitesse D (taille 6).
    Real_t D[6];

    // 单元内节点坐标
    // Coordonnées nodales dans l’élément.
    Real_t x_local[8];
    Real_t y_local[8];
    Real_t z_local[8];

    // 单元内节点速度
    // Vitesses nodales locales.
    Real_t xd_local[8];
    Real_t yd_local[8];
    Real_t zd_local[8];

    // Jacobian 行列式
    // Déterminant du jacobien.
    Real_t detJ = Real_t(0.0);

    // 单元体积与相对体积
    // Volume de l’élément et volume relatif.
    Real_t volume;
    Real_t relativeVolume;

    // 单元到全局节点编号映射
    // Mappage élément→nœud global.
    const Index_t *const elemToNode = domain.nodelist(k);

    // 收集本单元的 8 个节点坐标
    // Récupère les coordonnées des 8 nœuds de l’élément.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // 计算单元体积
    // Calcule le volume de l’élément.
    volume = CalcElemVolume(x_local, y_local, z_local);

    // 计算体积比 v_new = volume / volo
    // Calcule le volume relatif v_new = volume / volo.
    relativeVolume = volume / domain.volo(k);
    domain.vnew(k) = relativeVolume;

    // 记录体积增量 delv = v_new - v_old
    // Stocke la variation volumique delv = v_new - v_old.
    domain.delv(k) = relativeVolume - domain.v(k);

    // 计算单元的特征长度
    // Calcule la longueur caractéristique de l’élément.
    domain.arealg(k) =
        CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

    // 读取节点速度到局部数组
    // Charge les vitesses nodales dans les tableaux locaux.
    for (Index_t lnode = 0; lnode < 8; ++lnode) {
      Index_t gnode = elemToNode[lnode];
      xd_local[lnode] = domain.xd(gnode);
      yd_local[lnode] = domain.yd(gnode);
      zd_local[lnode] = domain.zd(gnode);
    }

    // 半步时间步长 dt/2
    // Pas de temps demi‐étape dt/2.
    Real_t dt2 = Real_t(0.5) * deltaTime;

    // 退回半步位置（中心差分方法需要）
    // Recul d’une demi‐étape temporelle (nécessaire au schéma en différences centrées).
    for (Index_t j = 0; j < 8; ++j) {
      x_local[j] -= dt2 * xd_local[j];
      y_local[j] -= dt2 * yd_local[j];
      z_local[j] -= dt2 * zd_local[j];
    }

    // 基于位置计算形函数导数 B 与 Jacobian detJ
    // Calcule les dérivées des fonctions de forme B et le déterminant detJ.
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);

    // 计算单元速度梯度 D（包含 ∂vx/∂x,… 及对称项）
    // Calcule le gradient de vitesse D (incluant ∂vx/∂x, … et les termes symétriques).
    CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);

    // 将速度梯度写回 domain
    // Stocke le gradient de vitesse dans le domain.
    domain.dxx(k) = D[0];
    domain.dyy(k) = D[1];
    domain.dzz(k) = D[2];
  });
}

// 计算拉格朗日单元量（体积变化 + 无量纲速度散度）
// Calcule les grandeurs lagrangiennes des éléments (variation volumique + divergence de vitesse).
static inline void CalcLagrangeElements(Domain &domain) {
  Index_t numElem = domain.numElem();  
  // 单元数量
  // Nombre d’éléments.

  if (numElem > 0) {
    const Real_t deltatime = domain.deltatime();
    // 时间步长
    // Pas de temps.

    domain.AllocateStrains(numElem);
    // 分配存储应变的数组
    // Alloue les tableaux nécessaires pour les déformations.

    CalcKinematicsForElems(domain, deltatime, numElem);
    // 计算体积、体积变化、速度梯度
    // Calcule volume, variation volumique et gradient de vitesse.

    Kokkos::parallel_for(numElem, [=](const int k) {
      Real_t vdov = domain.dxx(k) + domain.dyy(k) + domain.dzz(k);
      // 速度散度（迹）
      // Divergence de vitesse (trace du gradient).

      Real_t vdovthird = vdov / Real_t(3.0);
      // 平均散度（各方向扣除同样量）
      // Divergence moyenne (soustraite également dans chaque direction).

      domain.vdov(k) = vdov;
      // 存储散度
      // Stocke la divergence.

      domain.dxx(k) -= vdovthird;
      domain.dyy(k) -= vdovthird;
      domain.dzz(k) -= vdovthird;
      // 去除体积部分得到纯形变梯度
      // Enlève la partie volumique pour obtenir la déformation pure.

      if (domain.vnew(k) <= Real_t(0.0)) {
        // 检查是否出现单元体积崩塌
        // Vérifie l’effondrement volumique de l’élément.
      #if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError);
      #else
        exit(VolumeError);
      #endif
      }
    });

    domain.DeallocateStrains();
    // 释放应变内存
    // Libère la mémoire associée aux déformations.
  }
}

// 计算 Q 单调性限制器所需的梯度（对应 X, Y, Z 三个方向）
// Calcule les gradients nécessaires pour le limiteur monotone Q (directions X, Y, Z).
static inline void CalcMonotonicQGradientsForElems(Domain &domain) {
  Index_t numElem = domain.numElem();
  // 单元数量
  // Nombre d’éléments.

  Kokkos::parallel_for("CalcMonotonicQGradientsForElems", numElem,
                       KOKKOS_LAMBDA(const int i) {

    const Real_t ptiny = Real_t(1.e-36);
    // 防止除零的小量
    // Petite valeur pour éviter la division par zéro.

    Real_t ax, ay, az;
    Real_t dxv, dyv, dzv;

    const Index_t *elemToNode = domain.nodelist(i);
    // 单元到节点的映射
    // Mappage élément → nœuds.

    Index_t n0 = elemToNode[0];
    Index_t n1 = elemToNode[1];
    Index_t n2 = elemToNode[2];
    Index_t n3 = elemToNode[3];
    Index_t n4 = elemToNode[4];
    Index_t n5 = elemToNode[5];
    Index_t n6 = elemToNode[6];
    Index_t n7 = elemToNode[7];
    // 八个节点的编号
    // Indices des huit nœuds.

    // 读取节点坐标
    // Charge les coordonnées nodales.
    Real_t x0 = domain.x(n0); Real_t x1 = domain.x(n1);
    Real_t x2 = domain.x(n2); Real_t x3 = domain.x(n3);
    Real_t x4 = domain.x(n4); Real_t x5 = domain.x(n5);
    Real_t x6 = domain.x(n6); Real_t x7 = domain.x(n7);

    Real_t y0 = domain.y(n0); Real_t y1 = domain.y(n1);
    Real_t y2 = domain.y(n2); Real_t y3 = domain.y(n3);
    Real_t y4 = domain.y(n4); Real_t y5 = domain.y(n5);
    Real_t y6 = domain.y(n6); Real_t y7 = domain.y(n7);

    Real_t z0 = domain.z(n0); Real_t z1 = domain.z(n1);
    Real_t z2 = domain.z(n2); Real_t z3 = domain.z(n3);
    Real_t z4 = domain.z(n4); Real_t z5 = domain.z(n5);
    Real_t z6 = domain.z(n6); Real_t z7 = domain.z(n7);

    // 读取节点速度
    // Charge les vitesses nodales.
    Real_t xv0 = domain.xd(n0); Real_t xv1 = domain.xd(n1);
    Real_t xv2 = domain.xd(n2); Real_t xv3 = domain.xd(n3);
    Real_t xv4 = domain.xd(n4); Real_t xv5 = domain.xd(n5);
    Real_t xv6 = domain.xd(n6); Real_t xv7 = domain.xd(n7);

    Real_t yv0 = domain.yd(n0); Real_t yv1 = domain.yd(n1);
    Real_t yv2 = domain.yd(n2); Real_t yv3 = domain.yd(n3);
    Real_t yv4 = domain.yd(n4); Real_t yv5 = domain.yd(n5);
    Real_t yv6 = domain.yd(n6); Real_t yv7 = domain.yd(n7);

    Real_t zv0 = domain.zd(n0); Real_t zv1 = domain.zd(n1);
    Real_t zv2 = domain.zd(n2); Real_t zv3 = domain.zd(n3);
    Real_t zv4 = domain.zd(n4); Real_t zv5 = domain.zd(n5);
    Real_t zv6 = domain.zd(n6); Real_t zv7 = domain.zd(n7);

    // 当前体积
    // Volume actuel.
    Real_t vol = domain.volo(i) * domain.vnew(i);

    Real_t norm = Real_t(1.0) / (vol + ptiny);
    // 标准化因子（方向向量正规化）
    // Facteur de normalisation.

    // 计算 zeta 方向几何向量
    // Calcule les vecteurs géométriques pour la direction zeta
    Real_t dxj = Real_t(-0.25) * ((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
    Real_t dyj = Real_t(-0.25) * ((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
    Real_t dzj = Real_t(-0.25) * ((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));

    Real_t dxi = Real_t(0.25) * ((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
    Real_t dyi = Real_t(0.25) * ((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
    Real_t dzi = Real_t(0.25) * ((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));

    Real_t dxk = Real_t(0.25) * ((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
    Real_t dyk = Real_t(0.25) * ((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
    Real_t dzk = Real_t(0.25) * ((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));

    // 计算法向量 (ax, ay, az)
    // Calcule le vecteur normal (ax, ay, az).
    ax = dyi * dzj - dzi * dyj;
    ay = dzi * dxj - dxi * dzj;
    az = dxi * dyj - dyi * dxj;

    // 计算 delx_zeta
    // Calcule delx_zeta.
    domain.delx_zeta(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm; ay *= norm; az *= norm;
    // 正规化
    // Normalisation.

    dxv = Real_t(0.25) * ((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
    dyv = Real_t(0.25) * ((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
    dzv = Real_t(0.25) * ((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));

    domain.delv_zeta(i) = ax * dxv + ay * dyv + az * dzv;
    // zeta 方向速度梯度
    // Gradient de vitesse en direction zeta.

    // xi 方向
    ax = dyj * dzk - dzj * dyk;
    ay = dzj * dxk - dxj * dzk;
    az = dxj * dyk - dyj * dxk;

    domain.delx_xi(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm; ay *= norm; az *= norm;

    dxv = Real_t(0.25) * ((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
    dyv = Real_t(0.25) * ((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
    dzv = Real_t(0.25) * ((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));

    domain.delv_xi(i) = ax * dxv + ay * dyv + az * dzv;

    // eta 方向
    ax = dyk * dzi - dzk * dyi;
    ay = dzk * dxi - dxk * dzi;
    az = dxk * dyi - dyk * dxi;

    domain.delx_eta(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm; ay *= norm; az *= norm;

    dxv = Real_t(-0.25) * ((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
    dyv = Real_t(-0.25) * ((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
    dzv = Real_t(-0.25) * ((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));

    domain.delv_eta(i) = ax * dxv + ay * dyv + az * dzv;
  });
}

// 对区域 r 内的单元计算 Q 单调性限制器
// Calcule le limiteur monotone Q pour les éléments de la région r.
static inline void CalcMonotonicQRegionForElems(Domain &domain, Int_t r,
                                                Real_t ptiny) {

  Real_t monoq_limiter_mult = domain.monoq_limiter_mult();
  // 单调性限制器倍率
  // Multiplicateur du limiteur monotone.

  Real_t monoq_max_slope = domain.monoq_max_slope();
  // 最大允许斜率
  // Pente maximale autorisée.

  Real_t qlc_monoq = domain.qlc_monoq();
  // 线性 Q 系数
  // Coefficient Q linéaire.

  Real_t qqc_monoq = domain.qqc_monoq();
  // 二次 Q 系数
  // Coefficient Q quadratique.

  Kokkos::parallel_for("CalcMonotonicQRegionForElems", domain.regElemSize(r),
                       KOKKOS_LAMBDA(const int i) {

    Index_t ielem = domain.regElemlist(r, i);
    // 取区域 r 中的第 i 个单元索引
    // Récupère l’indice du i-ème élément de la région r.

    Real_t qlin, qquad;
    // 线性项与二次项
    // Terme linéaire et terme quadratique.

    Real_t phixi, phieta, phizeta;
    // 三个方向的限制器系数
    // Coefficients du limiteur dans les trois directions.

    Int_t bcMask = domain.elemBC(ielem);
    // 边界条件 mask
    // Masque des conditions aux limites.

    Real_t delvm = 0.0, delvp = 0.0;
    // 左右方向的速度梯度增量
    // Incréments du gradient de vitesse (gauche et droite).

    Real_t norm = Real_t(1.) / (domain.delv_xi(ielem) + ptiny);
    // 归一化因子，避免除零
    // Facteur de normalisation, évite division par zéro.

    // 处理 XI 方向的 delvm
    // Traitement de delvm en direction XI
    switch (bcMask & XI_M) {
    case XI_M_COMM:
    case 0:
      delvm = domain.delv_xi(domain.lxim(ielem));
      // 通信或内部节点：取邻居 delv_xi
      // Élément interne ou en communication : utilise le voisin.
      break;

    case XI_M_SYMM:
      delvm = domain.delv_xi(ielem);
      // 对称边界：自身梯度为零，使用本单元
      // Condition symétrique : dérivée nulle → utilise l’élément courant.
      break;

    case XI_M_FREE:
      delvm = Real_t(0.0);
      // 自由边界：梯度设为 0
      // Condition libre : gradient nul.
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      // 错误情况：打印错误信息（法语结构保持）  
      // Cas invalide : imprime un message d’erreur.
      delvm = 0;
      break;
    }

    // 处理 XI 方向的 delvp
    // Traitement de delvp en direction XI
    switch (bcMask & XI_P) {
    case XI_P_COMM:
    case 0:
      delvp = domain.delv_xi(domain.lxip(ielem));
      // 内部或通信：取 +xi 方向邻居
      // Interne/communication : voisin en direction +xi.
      break;

    case XI_P_SYMM:
      delvp = domain.delv_xi(ielem);
      // 对称：自身梯度
      // Symétrique : utilise l’élément lui-même.
      break;

    case XI_P_FREE:
      delvp = Real_t(0.0);
      // 自由边界
      // Frontière libre.
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      // 错误情况
      // Erreur.
      delvp = 0;
      break;
    }

    delvm = delvm * norm;
    delvp = delvp * norm;
    // 应用归一化
    // Applique la normalisation.

    phixi = Real_t(.5) * (delvm + delvp);
    // 初始限制器值为平均梯度
    // Valeur initiale = moyenne des gradients.

    delvm *= monoq_limiter_mult;
    delvp *= monoq_limiter_mult;
    // 扩大限制器范围
    // Amplifie la zone d’action du limiteur.

    if (delvm < phixi) phixi = delvm;
    if (delvp < phixi) phixi = delvp;
    // phixi 不能比两侧最小梯度更大
    // phixi ne doit pas dépasser le minimum des deux gradients.

    if (phixi < Real_t(0.)) phixi = Real_t(0.);
    // 限制器不能为负
    // Le limiteur ne peut pas être négatif.

    if (phixi > monoq_max_slope) phixi = monoq_max_slope;
    // 限制器不能超过最大斜率
    // Le limiteur ne peut pas dépasser la pente maximale.

    // Ci-dessous traitement pour la direction ETA
    norm = Real_t(1.) / (domain.delv_eta(ielem) + ptiny);
    // 归一化因子
    // Facteur de normalisation.

    switch (bcMask & ETA_M) {
    case ETA_M_COMM:
    case 0:
      delvm = domain.delv_eta(domain.letam(ielem));
      break;

    case ETA_M_SYMM:
      delvm = domain.delv_eta(ielem);
      break;

    case ETA_M_FREE:
      delvm = Real_t(0.0);
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      delvm = 0;
      break;
    }

    switch (bcMask & ETA_P) {
    case ETA_P_COMM:
    case 0:
      delvp = domain.delv_eta(domain.letap(ielem));
      break;

    case ETA_P_SYMM:
      delvp = domain.delv_eta(ielem);
      break;

    case ETA_P_FREE:
      delvp = Real_t(0.0);
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      delvp = 0;
      break;
    }

        delvm = delvm * norm;
    // 左向梯度归一化
    // Normalisation du gradient vers la direction négative.

    delvp = delvp * norm;
    // 右向梯度归一化
    // Normalisation du gradient vers la direction positive.

    phieta = Real_t(.5) * (delvm + delvp);
    // 初始 eta 方向限制器取两侧梯度平均
    // Le limiteur initial en direction eta est la moyenne des deux gradients.

    delvm *= monoq_limiter_mult;
    // 扩大量程（左）
    // Étend la plage d’action du limiteur (gauche).

    delvp *= monoq_limiter_mult;
    // 扩大量程（右）
    // Étend la plage d’action du limiteur (droite).

    if (delvm < phieta)
      phieta = delvm;
      // 限制器不超过最小梯度（左）
      // Le limiteur ne dépasse pas le gradient minimum (gauche).

    if (delvp < phieta)
      phieta = delvp;
      // 限制器不超过最小梯度（右）
      // Le limiteur ne dépasse pas le gradient minimum (droite).

    if (phieta < Real_t(0.))
      phieta = Real_t(0.);
      // 不能为负
      // Ne peut pas être négatif.

    if (phieta > monoq_max_slope)
      phieta = monoq_max_slope;
      // 不能超过最大斜率
      // Ne peut pas dépasser la pente maximale.

    norm = Real_t(1.) / (domain.delv_zeta(ielem) + ptiny);
    // zeta 方向归一化因子
    // Facteur de normalisation pour la direction zeta.

    switch (bcMask & ZETA_M) {
    case ZETA_M_COMM:
    case 0:
      delvm = domain.delv_zeta(domain.lzetam(ielem));
      // 内部/通信：取负向邻居
      // Interne/communication : utilise le voisin négatif.
      break;

    case ZETA_M_SYMM:
      delvm = domain.delv_zeta(ielem);
      // 对称边界：自身梯度
      // Symétrique : utilise le gradient local.
      break;

    case ZETA_M_FREE:
      delvm = Real_t(0.0);
      // 自由边界：梯度为零
      // Libre : gradient nul.
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      // 错误情况输出（法语保持结构）
      // Impression d’erreur pour un cas non géré.
      delvm = 0;
      break;
    }

    switch (bcMask & ZETA_P) {
    case ZETA_P_COMM:
    case 0:
      delvp = domain.delv_zeta(domain.lzetap(ielem));
      // 正向邻居
      // Voisin dans la direction positive.
      break;

    case ZETA_P_SYMM:
      delvp = domain.delv_zeta(ielem);
      // 对称边界
      // Condition symétrique.
      break;

    case ZETA_P_FREE:
      delvp = Real_t(0.0);
      // 自由边界
      // Frontière libre.
      break;

    default:
      fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
      // 错误输出
      // Impression d’erreur.
      delvp = 0;
      break;
    }

    delvm = delvm * norm;
    // 左侧梯度归一化
    // Normalisation du gradient négatif.

    delvp = delvp * norm;
    // 右侧梯度归一化
    // Normalisation du gradient positif.

    phizeta = Real_t(.5) * (delvm + delvp);
    // zeta 方向限制器初值
    // Valeur initiale du limiteur en direction zeta.

    delvm *= monoq_limiter_mult;
    // 扩大量程（左）
    // Extension de plage (gauche).

    delvp *= monoq_limiter_mult;
    // 扩大量程（右）
    // Extension de plage (droite).

    if (delvm < phizeta)
      phizeta = delvm;
      // 限制最小值（左）
      // Limite au minimum (gauche).

    if (delvp < phizeta)
      phizeta = delvp;
      // 限制最小值（右）
      // Limite au minimum (droite).

    if (phizeta < Real_t(0.))
      phizeta = Real_t(0.);
      // 非负约束
      // Contrainte de positivité.

    if (phizeta > monoq_max_slope)
      phizeta = monoq_max_slope;
      // 上界约束
      // Contrainte de pente maximale.

    if (domain.vdov(ielem) > Real_t(0.)) {
      qlin = Real_t(0.);
      qquad = Real_t(0.);
      // 体积膨胀时不产生粘性 Q
      // Pas de viscosité Q en cas d’expansion volumique.
    } else {

      Real_t delvxxi = domain.delv_xi(ielem) * domain.delx_xi(ielem);
      // xi 方向速度变化量
      // Variation de vitesse en direction xi.

      Real_t delvxeta = domain.delv_eta(ielem) * domain.delx_eta(ielem);
      // eta 方向速度变化量
      // Variation de vitesse en direction eta.

      Real_t delvxzeta = domain.delv_zeta(ielem) * domain.delx_zeta(ielem);
      // zeta 方向速度变化量
      // Variation de vitesse en direction zeta.

      if (delvxxi > Real_t(0.))
        delvxxi = Real_t(0.);
        // 正值不参与压缩 Q
        // Les valeurs positives ne contribuent pas à la compression.

      if (delvxeta > Real_t(0.))
        delvxeta = Real_t(0.);
        // 同上
        // Idem.

      if (delvxzeta > Real_t(0.))
        delvxzeta = Real_t(0.);
        // 同上
        // Idem.

      Real_t rho =
          domain.elemMass(ielem) /
          (domain.volo(ielem) * domain.vnew(ielem));
      // 当前密度
      // Densité actuelle.

      qlin = -qlc_monoq * rho *
             (delvxxi * (Real_t(1.) - phixi) +
              delvxeta * (Real_t(1.) - phieta) +
              delvxzeta * (Real_t(1.) - phizeta));
      // 线性人工粘性项
      // Terme linéaire de viscosité artificielle.

      qquad = qqc_monoq * rho *
             (delvxxi * delvxxi * (Real_t(1.) - phixi * phixi) +
              delvxeta * delvxeta * (Real_t(1.) - phieta * phieta) +
              delvxzeta * delvxzeta * (Real_t(1.) - phizeta * phizeta));
      // 二次人工粘性项
      // Terme quadratique de viscosité artificielle.
    }

    domain.qq(ielem) = qquad;
    // 存储二次 Q
    // Stocke le terme quadratique.

    domain.ql(ielem) = qlin;
    // 存储线性 Q
    // Stocke le terme linéaire.
    });
}

static inline void CalcMonotonicQForElems(Domain &domain) {
  const Real_t ptiny = Real_t(1.e-36);
  // 极小数防止除零
  // Très petite valeur pour éviter la division par zéro.

  for (Index_t r = 0; r < domain.numReg(); ++r) {
    // 遍历所有区域
    // Parcourt toutes les régions.

    if (domain.regElemSize(r) > 0) {
      // 仅处理含有单元的区域
      // Ne traite que les régions contenant des éléments.

      CalcMonotonicQRegionForElems(domain, r, ptiny);
      // 调用区域级 Q 单调性计算
      // Appelle le calcul monotone du Q pour la région.
    }
  }
}

static inline void CalcQForElems(Domain &domain) {
  Index_t numElem = domain.numElem();
  // 当前本地单元数量
  // Nombre d’éléments locaux.

  if (numElem != 0) {

    Int_t allElem = numElem +                             
                    2 * domain.sizeX() * domain.sizeY() + 
                    2 * domain.sizeX() * domain.sizeZ() + 
                    2 * domain.sizeY() * domain.sizeZ();
    // 总单元数量（包含 ghost 单元）
    // Nombre total d’éléments (y compris les éléments fantômes).

    domain.AllocateGradients(numElem, allElem);
    // 分配梯度存储空间
    // Alloue la mémoire pour les gradients.

#if USE_MPI
    CommRecv(domain, MSG_MONOQ, 3, domain.sizeX(), domain.sizeY(),
             domain.sizeZ(), true, true);
    // MPI 接收梯度相关 ghost 数据
    // MPI reçoit les données fantômes des gradients.
#endif

    CalcMonotonicQGradientsForElems(domain);
    // 计算单元梯度（delv_xi, eta, zeta）
    // Calcule les gradients des éléments.

#if USE_MPI
    Domain_member fieldData[3];
    // MPI 字段指针数组
    // Tableau des pointeurs vers les champs MPI.

    fieldData[0] = &Domain::delv_xi;
    fieldData[1] = &Domain::delv_eta;
    fieldData[2] = &Domain::delv_zeta;
    // 注册要通信的三个梯度字段
    // Enregistre les trois champs de gradients à communiquer.

    CommSend(domain, MSG_MONOQ, 3, fieldData, domain.sizeX(), domain.sizeY(),
             domain.sizeZ(), true, true);
    // MPI 发送梯度到邻居
    // MPI envoie les gradients aux voisins.

    CommMonoQ(domain);
    // 完成 Q 的边界交换
    // Finalise l’échange de Q.
#endif

    CalcMonotonicQForElems(domain);
    // 对所有区域计算 Q（单调限制器）
    // Calcule le Q monotone pour tous les éléments.

    domain.DeallocateGradients();
    // 释放梯度内存
    // Libère la mémoire des gradients.

    Index_t idx = -1;
    // 默认没有触发 qstop
    // Par défaut, aucun qstop déclenché.

    typedef Kokkos::View<Index_t *> view_type_int;
    view_type_int idxfind("A", numElem);
    // 创建索引数组
    // Crée un tableau d’indices.

    Kokkos::parallel_for(numElem, [=](const Index_t i) {
      idxfind(i) = numElem;
      // 默认值为 numElem（即未触发）
      // Valeur par défaut = numElem (non déclenché).

      if (domain.q(i) > domain.qstop()) {
        idxfind(i) = i;
        // 找到 Q 过大的单元
        // Détecte un élément où Q dépasse le seuil.
      }
    });

    struct IdxMinFinder {
      int val;

      KOKKOS_INLINE_FUNCTION
      IdxMinFinder() : val(1000000000) {}
      // 初始化为很大的值
      // Initialise avec une très grande valeur.

      KOKKOS_INLINE_FUNCTION
      IdxMinFinder(const int &val_) : val(val_) {}
      // 从给定值构造
      // Construit à partir d’une valeur donnée.

      KOKKOS_INLINE_FUNCTION
      IdxMinFinder(const IdxMinFinder &src) : val(src.val) {}
      // 拷贝构造
      // Constructeur par copie.

      KOKKOS_INLINE_FUNCTION
      void operator+=(IdxMinFinder &src) {
        if (src.val < val) {
          val = src.val;
          // 取最小值
          // Prend la valeur minimale.
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator+=(const volatile IdxMinFinder &src) volatile {
        if (src.val < val) {
          val = src.val;
          // volatile 版本：同样取最小值
          // Version volatile : prend aussi la valeur minimale.
        }
      }
    };

    IdxMinFinder result;
    // reduction 结果
    // Résultat de la réduction.

    Kokkos::parallel_reduce(numElem,
                            KOKKOS_LAMBDA(const int &i, IdxMinFinder &minf) {
                              IdxMinFinder tmp(idxfind(i));
                              minf += tmp;
                              // 使用最小值规约
                              // Réduction par minimum.
                            },
                            result);

    idx = result.val;
    // 得到全局最小索引
    // Obtient l’indice minimal global.

    if (idx == numElem) {
      idx = -1;
      // 若没有任何单元触发，则 idx = -1
      // Si aucun élément n’a déclenché, idx = -1.
    }

    if (idx >= 0) {
      // 如果发现不满足 qstop 的单元，终止程序
      // Si un élément dépasse qstop, arrêt du programme.

#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, QStopError);
      // MPI 模式下终止所有进程
      // Arrêt global en MPI.
#else
      exit(QStopError);
      // 非 MPI：退出程序
      // Sans MPI : quitte le programme.
#endif
    }
  }
}

static inline void CalcPressureForElems(Real_t *p_new, Real_t *bvc,
                                        Real_t *pbvc, Real_t *e_old,
                                        Real_t *compression, Real_t *vnewc,
                                        Real_t pmin, Real_t p_cut,
                                        Real_t eosvmax, Index_t length,
                                        Index_t *regElemList) {

  Kokkos::parallel_for("CalcPressureForElems A", length,
                       KOKKOS_LAMBDA(const int i) {
    Real_t c1s = Real_t(2.0) / Real_t(3.0);
    // 计算体模量系数 2/3
    // Calcule le coefficient volumique 2/3.

    bvc[i] = c1s * (compression[i] + Real_t(1.));
    // bvc = 2/3 × (压缩率 + 1)
    // bvc = 2/3 × (compression + 1).

    pbvc[i] = c1s;
    // pbvc 恒为 2/3
    // pbvc vaut toujours 2/3.
  });

  Kokkos::parallel_for("CalcPressureForElems B", length,
                       KOKKOS_LAMBDA(const int i) {
    Index_t ielem = regElemList[i];
    // 获取真实单元 id
    // Récupère l'identifiant réel de l’élément.

    p_new[i] = bvc[i] * e_old[i];
    // 新压力 = bvc × 内能
    // Nouvelle pression = bvc × énergie interne.

    if (FABS(p_new[i]) < p_cut)
      p_new[i] = Real_t(0.0);
    // 小于阈值的压力截断为 0
    // Pression trop faible → coupée à zéro.

    if (vnewc[ielem] >= eosvmax)
      p_new[i] = Real_t(0.0);
    // 若体积过大，压力强制为 0
    // Si volume trop grand, pression annulée.

    if (p_new[i] < pmin)
      p_new[i] = pmin;
    // 压力不能低于最小值
    // Pression limitée par un minimum.
  });
}

static inline void
CalcEnergyForElems(Real_t *p_new, Real_t *e_new, Real_t *q_new, Real_t *bvc,
                   Real_t *pbvc, Real_t *p_old, Real_t *e_old, Real_t *q_old,
                   Real_t *compression, Real_t *compHalfStep, Real_t *vnewc,
                   Real_t *work, Real_t *delvc, Real_t pmin, Real_t p_cut,
                   Real_t e_cut, Real_t q_cut, Real_t emin, Real_t *qq_old,
                   Real_t *ql_old, Real_t rho0, Real_t eosvmax, Index_t length,
                   Index_t *regElemList) {

  Real_t *pHalfStep = Allocate<Real_t>(length);
  // 分配半步压力数组
  // Alloue le tableau de pression au demi-pas.

  Kokkos::parallel_for("CalcEnergyForElems A", length,
                       KOKKOS_LAMBDA(const int i) {

    e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i]) +
               Real_t(0.5) * work[i];
    // 基于体积变化 + 做功计算半步能量
    // Énergie demi-pas = ancienne énergie − 1/2 ΔV × (p + q) + 1/2 × travail.

    if (e_new[i] < emin) {
      e_new[i] = emin;
    }
    // 内能不能低于最小值
    // L’énergie interne est bornée inférieurement.
  });

  CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc, pmin,
                       p_cut, eosvmax, length, regElemList);
  // 使用半步内能求半步压力
  // Calcule la pression au demi-pas à partir de l’énergie demi-pas.

  Kokkos::parallel_for("CalcEnergyForElems B", length,
                       KOKKOS_LAMBDA(const int i) {

    Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]);
    // 半步体积估计
    // Volume estimé au demi-pas.

    if (delvc[i] > Real_t(0.)) {
      q_new[i] = Real_t(0.);
      // 若体积增加，则无人工粘性
      // Si expansion → pas de viscosité artificielle.
    } else {
      Real_t ssc =
          (pbvc[i] * e_new[i] + vhalf * vhalf * bvc[i] * pHalfStep[i]) / rho0;
      // 声速平方（未取 sqrt 前）
      // Carré de la vitesse du son (avant racine).

      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = SQRT(ssc);
      }
      // 避免 sqrt(0)，并求声速
      // Évite sqrt(0) et calcule la vitesse du son.

      q_new[i] = (ssc * ql_old[i] + qq_old[i]);
      // 更新人工粘性
      // Met à jour la viscosité artificielle.
    }

    e_new[i] =
        e_new[i] +
        Real_t(0.5) * delvc[i] * (Real_t(3.0) * (p_old[i] + q_old[i]) -
                                  Real_t(4.0) * (pHalfStep[i] + q_new[i]));
    // 完整第二步能量修正
    // Correction complète de l’énergie interne.
  });

  Kokkos::parallel_for("CalcEnergyForElems C", length,
                       KOKKOS_LAMBDA(const int i) {
    e_new[i] += Real_t(0.5) * work[i];
    // 加上后半步做功
    // Ajoute la seconde moitié du travail.

    if (FABS(e_new[i]) < e_cut) {
      e_new[i] = Real_t(0.);
    }
    // 小能量截断
    // Coupure des petites valeurs.

    if (e_new[i] < emin) {
      e_new[i] = emin;
    }
    // 能量不能低于最小值
    // L’énergie interne est bornée.
  });

  CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut,
                       eosvmax, length, regElemList);
  // 使用最终内能重新计算压力
  // Recalcule la pression après mise à jour finale de l’énergie.

  Kokkos::parallel_for("CalcEnergyForElems D", length,
                       KOKKOS_LAMBDA(const int i) {

    const Real_t sixth = Real_t(1.0) / Real_t(6.0);
    // 六分之一系数
    // Coefficient un sixième.

    Index_t ielem = regElemList[i];
    // 单元 id
    // Identifiant réel de l’élément.

    Real_t q_tilde;

    if (delvc[i] > Real_t(0.)) {
      q_tilde = Real_t(0.);
      // 体积增加 → 无人工粘性
      // Expansion → pas de viscosité artificielle.
    } else {
      Real_t ssc = (pbvc[i] * e_new[i] +
                    vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i]) /
                   rho0;

      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = SQRT(ssc);
      }

      q_tilde = (ssc * ql_old[i] + qq_old[i]);
      // 新一轮人工粘性估计
      // Nouvelle estimation de la viscosité artificielle.
    }

    e_new[i] =
        e_new[i] -
        (Real_t(7.0) * (p_old[i] + q_old[i]) -
         Real_t(8.0) * (pHalfStep[i] + q_new[i]) + (p_new[i] + q_tilde)) *
            delvc[i] * sixth;
    // 完整三段式内能更新公式
    // Formule complète d’intégration en trois étapes de l’énergie interne.

    if (FABS(e_new[i]) < e_cut) {
      e_new[i] = Real_t(0.);
    }
    if (e_new[i] < emin) {
      e_new[i] = emin;
    }
  });

  CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut,
                       eosvmax, length, regElemList);
  // 最终再次更新压力，确保与最终内能一致
  // Met à jour la pression pour cohérence finale avec l’énergie.

  Kokkos::parallel_for("CalcEnergyForElems E", length,
                       KOKKOS_LAMBDA(const int i) {
    Index_t ielem = regElemList[i];

    if (delvc[i] <= Real_t(0.)) {
      // 仅在压缩时更新人工粘性
      // Viscosité artificielle mise à jour seulement en compression.

      Real_t ssc = (pbvc[i] * e_new[i] +
                    vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i]) /
                   rho0;

      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = SQRT(ssc);
      }

      q_new[i] = (ssc * ql_old[i] + qq_old[i]);

      if (FABS(q_new[i]) < q_cut)
        q_new[i] = Real_t(0.);
      // 小人工粘性截断为 0
      // Coupure des petites valeurs de q.
    }
  });

  Release(&pHalfStep);
  // 释放中间数组
  // Libère le tableau intermédiaire.

  return;
}

static inline void CalcSoundSpeedForElems(Domain &domain, Real_t *vnewc,
                                          Real_t rho0, Real_t *enewc,
                                          Real_t *pnewc, Real_t *pbvc,
                                          Real_t *bvc, Real_t ss4o3,
                                          Index_t len, Index_t *regElemList) {
  Kokkos::parallel_for("CalcSoundSpeedForElems", len,
                       KOKKOS_LAMBDA(const int i) {
    Index_t ielem = regElemList[i];
    // 获取真实单元索引
    // Récupère l’indice réel de l’élément.

    Real_t ssTmp =
        (pbvc[i] * enewc[i] + vnewc[ielem] * vnewc[ielem] * bvc[i] * pnewc[i]) /
        rho0;
    // 声速平方（未开方）= (pbvc·e + v²·bvc·p) / 密度
    // Carré de la vitesse du son (avant racine) = (pbvc·e + v²·bvc·p) / densité.

    if (ssTmp <= Real_t(.1111111e-36)) {
      ssTmp = Real_t(.3333333e-18);
      // 避免声速为零 → 设定极小正值
      // Pour éviter une vitesse nulle → valeur minimale positive.
    } else {
      ssTmp = SQRT(ssTmp);
      // 取平方根得到声速
      // Prend la racine carrée pour obtenir la vitesse du son.
    }

    domain.ss(ielem) = ssTmp;
    // 写入结果到域变量
    // Enregistre la vitesse du son dans le domaine.
  });
}

static inline void EvalEOSForElems(Domain &domain, Real_t *vnewc,
                                   Int_t numElemReg, Index_t *regElemList,
                                   Int_t rep) {

  Real_t e_cut = domain.e_cut();
  // 能量截断阈值
  // Seuil de coupure pour l’énergie.

  Real_t p_cut = domain.p_cut();
  // 压力截断阈值
  // Seuil de coupure pour la pression.

  Real_t ss4o3 = domain.ss4o3();
  // 4/3 的声速参数
  // Paramètre de vitesse du son (4/3).

  Real_t q_cut = domain.q_cut();
  // 人工粘性 q 的截断阈值
  // Seuil de coupure pour la viscosité artificielle q.

  Real_t eosvmax = domain.eosvmax();
  Real_t eosvmin = domain.eosvmin();
  // 允许体积的最大值和最小值
  // Volumes minimum et maximum autorisés.

  Real_t pmin = domain.pmin();
  Real_t emin = domain.emin();
  // 最小压力和最小能量
  // Pression minimale et énergie minimale.

  Real_t rho0 = domain.refdens();
  // 参考密度
  // Densité de référence.


  // 分配局部数组（每个区域独立）
  // Alloue les tableaux locaux (propres à la région).
  Real_t *e_old = Allocate<Real_t>(numElemReg);
  Real_t *delvc = Allocate<Real_t>(numElemReg);
  Real_t *p_old = Allocate<Real_t>(numElemReg);
  Real_t *q_old = Allocate<Real_t>(numElemReg);
  Real_t *compression = Allocate<Real_t>(numElemReg);
  Real_t *compHalfStep = Allocate<Real_t>(numElemReg);
  Real_t *qq_old = Allocate<Real_t>(numElemReg);
  Real_t *ql_old = Allocate<Real_t>(numElemReg);
  Real_t *work = Allocate<Real_t>(numElemReg);
  Real_t *p_new = Allocate<Real_t>(numElemReg);
  Real_t *e_new = Allocate<Real_t>(numElemReg);
  Real_t *q_new = Allocate<Real_t>(numElemReg);
  Real_t *bvc = Allocate<Real_t>(numElemReg);
  Real_t *pbvc = Allocate<Real_t>(numElemReg);


  for (Int_t j = 0; j < rep; j++) {
    // rep 次重复（某些 EOS 需要迭代）
    // Répète rep fois (certains EOS nécessitent des itérations).

    Kokkos::parallel_for("EvalEOSForElems A", numElemReg,
                         KOKKOS_LAMBDA(const int i) {
      Index_t ielem = regElemList[i];
      // 提取单元编号
      // Extrait l’indice réel de l’élément.

      e_old[i] = domain.e(ielem);
      delvc[i] = domain.delv(ielem);
      p_old[i] = domain.p(ielem);
      q_old[i] = domain.q(ielem);
      qq_old[i] = domain.qq(ielem);
      ql_old[i] = domain.ql(ielem);
      // 复制旧状态（能量·体积变化·压力·人工粘性）
      // Copie des anciens états (énergie, ΔV, pression, viscosité artificielle).
    });

    Kokkos::parallel_for("EvalEOSForElems B", numElemReg,
                         KOKKOS_LAMBDA(const int i) {
      Index_t ielem = regElemList[i];

      Real_t vchalf;
      compression[i] = Real_t(1.) / vnewc[ielem] - Real_t(1.);
      // 压缩率 = 1/V - 1
      // Compression = 1/V − 1.

      vchalf = vnewc[ielem] - delvc[i] * Real_t(.5);
      // 半步体积：Vₙ − ΔV/2
      // Volume demi-pas : Vₙ − ΔV/2.

      compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);
      // 半步压缩率 = 1/V_half − 1
      // Compression demi-pas = 1/V_half − 1.
    });

    if (eosvmin != Real_t(0.)) {
      Kokkos::parallel_for("EvalEOSForElems C", numElemReg,
                           KOKKOS_LAMBDA(const int i) {
        Index_t ielem = regElemList[i];
        if (vnewc[ielem] <= eosvmin) {
          // 如果体积过小，用整体压缩代替半步压缩
          // Si volume trop petit, utilise compression complète.
          compHalfStep[i] = compression[i];
        }
      });
    }

    if (eosvmax != Real_t(0.)) {
      Kokkos::parallel_for("EvalEOSForElems D", numElemReg,
                           KOKKOS_LAMBDA(const int i) {
        Index_t ielem = regElemList[i];
        if (vnewc[ielem] >= eosvmax) {
          // 如果体积过大：压力与压缩全部清零
          // Si volume trop grand : pression et compression annulées.
          p_old[i] = Real_t(0.);
          compression[i] = Real_t(0.);
          compHalfStep[i] = Real_t(0.);
        }
      });
    }

    Kokkos::parallel_for("EvalEOSForElems E", numElemReg,
                         KOKKOS_LAMBDA(const int i) {
      work[i] = Real_t(0.);
      // EOS 不考虑外功 → work = 0
      // L’EOS n’intègre pas de travail externe → work = 0.
    });


    // 计算新能量/压力/粘性（核心 EOS）
    // Calcul de l’énergie/pression/viscosité (cœur de l’EOS).
    CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old,
                       compression, compHalfStep, vnewc, work, delvc, pmin,
                       p_cut, e_cut, q_cut, emin, qq_old, ql_old, rho0,
                       eosvmax, numElemReg, regElemList);
  }

  Kokkos::parallel_for("EvalEOSForElems F", numElemReg,
                       KOKKOS_LAMBDA(const int i) {
    Index_t ielem = regElemList[i];
    // 写回域中
    // Écrit les résultats dans le domaine.

    domain.p(ielem) = p_new[i];
    domain.e(ielem) = e_new[i];
    domain.q(ielem) = q_new[i];
  });

  // 更新声速
  // Met à jour la vitesse du son.
  CalcSoundSpeedForElems(domain, vnewc, rho0, e_new, p_new,
                         pbvc, bvc, ss4o3, numElemReg, regElemList);

  // 释放内存
  // Libère la mémoire.
  Release(&pbvc);
  Release(&bvc);
  Release(&q_new);
  Release(&e_new);
  Release(&p_new);
  Release(&work);
  Release(&ql_old);
  Release(&qq_old);
  Release(&compHalfStep);
  Release(&compression);
  Release(&q_old);
  Release(&p_old);
  Release(&delvc);
  Release(&e_old);
}

static inline void ApplyMaterialPropertiesForElems(Domain &domain) {
  Index_t numElem = domain.numElem();
  // 读取单元数量
  // Lit le nombre d’éléments.

  if (numElem != 0) {
    // 若存在单元则继续
    // Continue seulement si des éléments existent.

    Real_t eosvmin = domain.eosvmin();
    Real_t eosvmax = domain.eosvmax();
    // EOS 支持的最小体积与最大体积
    // Volume minimal et maximal autorisés par l’EOS.

    Real_t *vnewc = Allocate<Real_t>(numElem);
    // 分配存放 vnew 的数组
    // Alloue un tableau pour vnew.

    Kokkos::parallel_for(
        "ApplyMaterialPropertiesForElems A", numElem,
        KOKKOS_LAMBDA(const int i) { vnewc[i] = domain.vnew(i); });
    // 复制每个单元最新体积到 vnewc
    // Copie les volumes mis à jour de chaque élément dans vnewc.

    if (eosvmin != Real_t(0.)) {
      // 若设置了体积下界，则进行限制
      // Si un volume minimal est défini, applique la limite.

      Kokkos::parallel_for("ApplyMaterialPropertiesForElems B", numElem,
                           KOKKOS_LAMBDA(const int i) {
        if (vnewc[i] < eosvmin)
          vnewc[i] = eosvmin;
        // 体积不能小于最小值
        // Le volume ne peut pas être inférieur au minimum.
      });
    }

    if (eosvmax != Real_t(0.)) {
      // 若设置了体积上界，则进行限制
      // Si un volume maximal est défini, applique la limite.

      Kokkos::parallel_for("ApplyMaterialPropertiesForElems C", numElem,
                           KOKKOS_LAMBDA(const int i) {
        if (vnewc[i] > eosvmax)
          vnewc[i] = eosvmax;
        // 体积不能大于最大值
        // Le volume ne peut pas dépasser le maximum.
      });
    }

    Kokkos::parallel_for(numElem, [=](const int i) {
      Real_t vc = domain.v(i);
      // 旧体积 vc
      // Ancien volume vc.

      if (eosvmin != Real_t(0.)) {
        if (vc < eosvmin)
          vc = eosvmin;
        // 检查旧体积是否过小
        // Vérifie si l’ancien volume est trop petit.
      }

      if (eosvmax != Real_t(0.)) {
        if (vc > eosvmax)
          vc = eosvmax;
        // 检查旧体积是否过大
        // Vérifie si l’ancien volume est trop grand.
      }

      if (vc <= 0.) {
        // 体积非法 → 终止程序
        // Volume non physique → arrêt du programme.
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
        exit(VolumeError);
#endif
      }
    });

    // 为每个区域调用 EOS（方程状态）
    // Appelle l’EOS (équation d’état) pour chaque région.

    for (Int_t r = 0; r < domain.numReg(); r++) {
      Index_t numElemReg = domain.regElemSize(r);
      Index_t *regElemList = domain.regElemlist(r);
      // 区域维度与区域元素列表
      // Taille de la région et liste des éléments de la région.

      Int_t rep;
      if (r < domain.numReg() / 2)
        rep = 1;
      // 前半区域：1 次迭代
      // Première moitié des régions : 1 itération.

      else if (r < (domain.numReg() - (domain.numReg() + 15) / 20))
        rep = 1 + domain.cost();
      // 中间区域：1 + cost 次迭代
      // Régions intermédiaires : 1 + cost itérations.

      else
        rep = 10 * (1 + domain.cost());
      // 最后 5% 区域：10×(1 + cost)
      // Derniers 5% : 10×(1 + cost).

      EvalEOSForElems(domain, vnewc, numElemReg, regElemList, rep);
      // 调用 EOS 更新压力、能量、人工粘性、声速
      // Appelle l’EOS pour mettre à jour pression, énergie, viscosité, vitesse du son.
    }

    Release(&vnewc);
    // 释放内存
    // Libère la mémoire.
  }
}

static inline void UpdateVolumesForElems(Domain &domain, Real_t v_cut,
                                         Index_t length) {
  if (length != 0) {
    // 若没有元素则无需更新
    // Aucun calcul si aucun élément.

    Kokkos::parallel_for("UpdateVolumesForElems", length,
                         KOKKOS_LAMBDA(const int i) {
      Real_t tmpV = domain.vnew(i);
      // 获取新体积
      // Récupère le nouveau volume.

      if (FABS(tmpV - Real_t(1.0)) < v_cut)
        tmpV = Real_t(1.0);
      // 若非常接近 1，则强制为 1（避免数值噪声）
      // Si très proche de 1, force à 1 pour réduire le bruit numérique.

      domain.v(i) = tmpV;
      // 更新到当前体积 v
      // Met à jour le volume courant v.
    });
  }

  return;
}

static inline void LagrangeElements(Domain &domain, Index_t numElem) {
  CalcLagrangeElements(domain);
  // 计算单元形变梯度与形状函数
  // Calcule les déformations et dérivées de forme des éléments.

  CalcQForElems(domain);
  // 计算人工粘性 Q
  // Calcule la viscosité artificielle Q.

  ApplyMaterialPropertiesForElems(domain);
  // 应用材料性质 + 调用 EOS（压力/能量/声速）
  // Applique les propriétés du matériau + appelle l’EOS.

  UpdateVolumesForElems(domain, domain.v_cut(), numElem);
  // 更新体积（归一化处理）
  // Met à jour les volumes (normalisation).
}

static inline void CalcCourantConstraintForElems(Domain &domain, Index_t length,
                                                 Index_t *regElemlist,
                                                 Real_t qqc,
                                                 Real_t &dtcourant) {
#if _OPENMP
  Index_t threads = omp_get_max_threads();
  static Index_t *courant_elem_per_thread;
  static Real_t *dtcourant_per_thread;
  static bool first = true;
  if (first) {
    courant_elem_per_thread = new Index_t[threads];
    dtcourant_per_thread = new Real_t[threads];
    first = false;
  }
  // OpenMP 多线程数组初始化
  // Initialisation des tableaux OpenMP pour le multithreading.
#else
  Index_t threads = 1;
  Index_t courant_elem_per_thread[1];
  Real_t dtcourant_per_thread[1];
  // 单线程执行模式
  // Mode d’exécution mono-thread.
#endif

  typedef Kokkos::View<Real_t *> view_real_t;
  view_real_t dtfV("A", length);
  // Kokkos View 存放 dtf 值
  // Vue Kokkos pour stocker les valeurs dtf.

  {
    Real_t qqc2 = Real_t(64.0) * qqc * qqc;
    // 预计算 64 * qqc^2
    // Pré-calcul de 64 * qqc^2.

    Real_t dtcourant_tmp = dtcourant;
    // 局部 Courant 时间步
    // Pas de temps Courant local.

    Index_t courant_elem = -1;
    // 最小时间步所属元素
    // Élément associé au pas de temps minimum.

#if _OPENMP
    Index_t thread_num = omp_get_thread_num();
    // 当前线程编号
    // Numéro du thread courant.
#else
    Index_t thread_num = 0;
    // 单线程：始终 0
    // Mono-thread : toujours 0.
#endif

    Kokkos::parallel_for(length, [=](const int i) {
      Index_t indx = regElemlist[i];
      // 当前元素索引
      // Indice de l’élément courant.

      Real_t dtf = domain.ss(indx) * domain.ss(indx);
      // 声速平方项
      // Carré de la vitesse du son.

      dtf = domain.ss(indx) * domain.ss(indx);

      if (domain.vdov(indx) < Real_t(0.)) {
        dtf = dtf +
              qqc2 * domain.arealg(indx) * domain.arealg(indx) *
                  domain.vdov(indx) * domain.vdov(indx);
        // 若速度散度 < 0：加入人工粘性项
        // Si divergence négative : ajoute le terme de viscosité artificielle.
      }

      dtf = SQRT(dtf);
      // 求平方根
      // Prend la racine carrée.

      dtfV(i) = domain.arealg(indx) / dtf;
      // 得到 dt 值候选
      // Produit le pas de temps candidat.
    });

    MinFinder result;
    // reduction 的结果结构
    // Structure pour stocker le résultat de la réduction.

    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int &i, MinFinder &minf) {
          MinFinder tmp(dtfV(i), i);
          // 构造最小值候选
          // Construit un candidat pour le minimum.

          Index_t indx = regElemlist[i];

          if (domain.vdov(indx) != Real_t(0.)) {
            minf += tmp;
            // 仅当速度散度 ≠ 0 才用于限制
            // Utilise seulement si divergence ≠ 0.
          }
        },
        result);

    dtcourant_tmp = result.val;
    // 得到最小 dt 值
    // Récupère le minimum du pas de temps.

    if (dtcourant_tmp > dtcourant) {
      dtcourant_tmp = dtcourant;
      // 不超过全局已有的 dtcourant
      // Ne dépasse pas le dtcourant global.
    }

    courant_elem = result.i;
    // 得到对应元素索引
    // Récupère l’indice de l’élément correspondant.

    dtcourant_per_thread[thread_num] = dtcourant_tmp;
    courant_elem_per_thread[thread_num] = courant_elem;
    // 保存该线程的最小值
    // Sauvegarde du minimum pour ce thread.
  }

  if (courant_elem_per_thread[0] != -1) {
    dtcourant = dtcourant_per_thread[0];
    // 单线程情况下直接更新
    // Mise à jour directe en mode mono-thread.
  }

  return;
}

static inline void CalcHydroConstraintForElems(Domain &domain, Index_t length,
                                               Index_t *regElemlist,
                                               Real_t dvovmax,
                                               Real_t &dthydro) {
#if _OPENMP
  Index_t threads = omp_get_max_threads();
  static Index_t *hydro_elem_per_thread;
  static Real_t *dthydro_per_thread;
  static bool first = true;
  if (first) {
    hydro_elem_per_thread = new Index_t[threads];
    dthydro_per_thread = new Real_t[threads];
    first = false;
  }
  // OpenMP：初始化线程局部数组
  // OpenMP : initialisation des tableaux locaux aux threads.
#else
  Index_t threads = 1;
  Index_t hydro_elem_per_thread[1];
  Real_t dthydro_per_thread[1];
  // 单线程模式
  // Mode mono-thread.
#endif

  typedef Kokkos::View<Real_t *> view_real_t;
  view_real_t dtdvovV("A", length);
  // Kokkos View 保存每个元素的时间步候选
  // Vue Kokkos pour stocker les pas de temps candidats.

  {
    Real_t dthydro_tmp = dthydro;
    // 初始水动力时间步
    // Pas de temps hydrodynamique initial.

    Index_t hydro_elem = -1;
    // 存最小时间步的元素索引
    // Indice de l’élément donnant le pas minimal.

#if _OPENMP
    Index_t thread_num = omp_get_thread_num();
    // 当前线程编号
    // Numéro du thread courant.
#else
    Index_t thread_num = 0;
    // 单线程始终为 0
    // Mono-thread : toujours 0.
#endif

    Kokkos::parallel_for(length, [=](const int i) {
      Index_t indx = regElemlist[i];
      // 当前元素索引
      // Indice de l’élément courant.

      if (domain.vdov(indx) != Real_t(0.)) {
        dtdvovV(i) = dvovmax / (FABS(domain.vdov(indx)) + Real_t(1.e-20));
        // 根据速度散度限制时间步
        // Limitation du pas de temps via la divergence de vitesse.
      }
    });

    MinFinder result;
    // reduction 用的最小值结构
    // Structure pour la réduction du minimum.

    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int &i, MinFinder &minf) {
          MinFinder tmp(dtdvovV(i), i);
          // 候选项
          // Candidat.

          Index_t indx = regElemlist[i];

          if (domain.vdov(indx) != Real_t(0.)) {
            minf += tmp;
            // 只有非零散度的元素才参与比较
            // Seulement les éléments avec divergence non nulle participent.
          }
        },
        result);

    if (result.val > dthydro) {
      result.val = dthydro;
      // 不超过当前全局限制
      // Ne doit pas dépasser la limite courante.
    }

    dthydro_per_thread[thread_num] = result.val;
    hydro_elem_per_thread[thread_num] = result.i;
    // 将当前线程的最小值写入保存区
    // Enregistre le minimum local du thread.
  }

  if (hydro_elem_per_thread[0] != -1) {
    dthydro = dthydro_per_thread[0];
    // 单线程：直接使用线程 0 的结果
    // Mono-thread : on utilise directement le résultat du thread 0.
  }

  return;
}



static inline void CalcTimeConstraintsForElems(Domain &domain) {

  domain.dtcourant() = 1.0e+20;
  // 初始化 Courant 限制为极大值
  // Initialise la contrainte de Courant à une valeur très grande.

  domain.dthydro() = 1.0e+20;
  // 初始化水动力限制为极大值
  // Initialise la contrainte hydrodynamique à une valeur très grande.

  for (Index_t r = 0; r < domain.numReg(); ++r) {
    CalcCourantConstraintForElems(domain, domain.regElemSize(r),
                                  domain.regElemlist(r), domain.qqc(),
                                  domain.dtcourant());
    // 计算 Courant 限制
    // Calcule la contrainte de Courant.

    CalcHydroConstraintForElems(domain, domain.regElemSize(r),
                                domain.regElemlist(r), domain.dvovmax(),
                                domain.dthydro());
    // 计算水动力限制
    // Calcule la contrainte hydrodynamique.
  }
}



static inline void LagrangeLeapFrog(Domain &domain) {
#ifdef SEDOV_SYNC_POS_VEL_LATE
  Domain_member fieldData[6];
  // 延迟同步：存储 x,y,z 与速度字段
  // Synchronisation tardive : stockage de x,y,z et vitesses.
#endif

  LagrangeNodal(domain);
  // 移动节点（位置 + 速度半步）
  // Mise à jour nodale (positions + demi-pas vitesse).

#ifdef SEDOV_SYNC_POS_VEL_LATE
#endif

  LagrangeElements(domain, domain.numElem());
  // 更新单元量（体积、几何量、应力等）
  // Mise à jour élémentaire (volumes, géométrie, contraintes).

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_LATE
  CommRecv(domain, MSG_SYNC_POS_VEL, 6, domain.sizeX() + 1, domain.sizeY() + 1,
           domain.sizeZ() + 1, false, false);
  // MPI 接收同步消息
  // Réception MPI pour la synchronisation.

  fieldData[0] = &Domain::x;
  fieldData[1] = &Domain::y;
  fieldData[2] = &Domain::z;
  fieldData[3] = &Domain::xd;
  fieldData[4] = &Domain::yd;
  fieldData[5] = &Domain::zd;
  // 指定需要同步的字段
  // Désigne les champs à synchroniser.

  CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData, domain.sizeX() + 1,
           domain.sizeY() + 1, domain.sizeZ() + 1, false, false);
  // MPI 发送同步消息
  // Envoi MPI de la synchronisation.
#endif
#endif

  CalcTimeConstraintsForElems(domain);
  // 计算全局时间步限制（Courant + 水动力）
  // Calcule les contraintes globales de pas de temps.

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_LATE
  CommSyncPosVel(domain);
  // 最终同步所有位置与速度
  // Synchronisation finale des positions et vitesses.
#endif
#endif
}

int main(int argc, char *argv[]) {
  Domain *locDom;  
  // 局部 Domain 指针  
  // Pointeur Domain local.

  Int_t numRanks;  
  // MPI 进程数量  
  // Nombre de processus MPI.

  Int_t myRank;  
  // 当前进程编号  
  // Rang du processus courant.

  struct cmdLineOpts opts;  
  // 命令行选项结构  
  // Structure des options de ligne de commande.

#if USE_MPI
  Domain_member fieldData;
  // MPI 同步的字段指针  
  // Pointeur vers les champs pour la synchronisation MPI.

  MPI_Init(&argc, &argv);
  // 初始化 MPI  
  // Initialisation MPI.

  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  // 获取 MPI 总进程数  
  // Récupère le nombre total de processus MPI.

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  // 获取当前 MPI 进程编号  
  // Récupère le rang du processus courant.
#else
  numRanks = 1;
  // 非 MPI 模式：固定为 1  
  // Mode non-MPI : valeur fixée à 1.

  myRank = 0;
  // 非 MPI 模式：只有 rank 0  
  // Mode non-MPI : seul le rang 0 existe.
#endif

  Kokkos::initialize();
  // 初始化 Kokkos  
  // Initialisation de Kokkos.

  opts.its = 9999999;  
  // 最大迭代次数  
  // Nombre maximal d’itérations.

  opts.nx = 30;  
  // 立方体边长（每维大小）  
  // Taille du cube (nombre de nœuds par dimension).

  opts.numReg = 11;  
  // 区域数量  
  // Nombre de régions.

  opts.numFiles = (int)(numRanks + 10) / 9;  
  // 可视化输出文件数量  
  // Nombre de fichiers pour la visualisation.

  opts.showProg = 0;  
  // 是否显示进度  
  // Affichage de la progression.

  opts.quiet = 0;  
  // 安静模式标志  
  // Mode silencieux.

  opts.viz = 0;  
  // 是否输出 VisIt 文件  
  // Génération des fichiers VisIt.

  opts.balance = 1;  
  // 区域平衡参数  
  // Paramètre d’équilibrage des régions.

  opts.cost = 1;  
  // 区域代价系数  
  // Coût relatif des régions.

  ParseCommandLineOptions(argc, argv, myRank, &opts);
  // 解析命令行参数  
  // Analyse des options de ligne de commande.

  if ((myRank == 0) && (opts.quiet == 0)) {
    printf("Running problem size %d^3 per domain until completion\n", opts.nx);
    // 打印每个域的计算规模  
    // Affiche la taille du problème par domaine.

    printf("Num processors: %d\n", numRanks);
    // 打印处理器数量  
    // Affiche le nombre de processeurs.

#if _OPENMP
    printf("Num threads: %d\n", omp_get_max_threads());
    // 打印 OpenMP 线程数  
    // Affiche le nombre de threads OpenMP.
#endif

    printf("Total number of elements: %lld\n\n",
           (long long int)(numRanks * opts.nx * opts.nx * opts.nx));
    // 打印总单元数  
    // Affiche le nombre total d’éléments.

    printf("To run other sizes, use -s <integer>.\n");
    // 用 -s 改变规模  
    // Modifier la taille avec -s.

    printf("To run a fixed number of iterations, use -i <integer>.\n");
    // 用 -i 控制迭代次数  
    // Fixer le nombre d’itérations avec -i.

    printf("To run a more or less balanced region set, use -b <integer>.\n");
    // 用 -b 控制区域平衡  
    // Ajuster l’équilibrage avec -b.

    printf("To change the relative costs of regions, use -c <integer>.\n");
    // 用 -c 设置区域代价  
    // Ajuster les coûts régionaux avec -c.

    printf("To print out progress, use -p\n");
    // 用 -p 显示运行进度  
    // Afficher la progression avec -p.

    printf("To write an output file for VisIt, use -v\n");
    // 用 -v 输出 VisIt 文件  
    // Générer les fichiers VisIt avec -v.

    printf("See help (-h) for more options\n\n");
    // 更多选项见 -h  
    // Voir -h pour plus d’options.
  }

  Int_t col, row, plane, side;  
  // 网格分解信息  
  // Informations de décomposition du maillage.

  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);
  // 初始化网格分解  
  // Initialisation de la décomposition du maillage.

  locDom = new Domain(numRanks, col, row, plane, opts.nx, side, opts.numReg,
                      opts.balance, opts.cost);
  // 创建 Domain 对象  
  // Création de l’objet Domain.

#if USE_MPI
  fieldData = &Domain::nodalMass;
  // 选择 nodalMass 字段用于同步  
  // Sélection du champ nodalMass pour la synchronisation.

  CommRecv(*locDom, MSG_COMM_SBN, 1, locDom->sizeX() + 1, locDom->sizeY() + 1,
           locDom->sizeZ() + 1, true, false);
  // 接收 ghost 数据  
  // Réception des données fantômes.

  CommSend(*locDom, MSG_COMM_SBN, 1, &fieldData, locDom->sizeX() + 1,
           locDom->sizeY() + 1, locDom->sizeZ() + 1, true, false);
  // 发送 ghost 数据  
  // Envoi des données fantômes.

  CommSBN(*locDom, 1, &fieldData);
  // 完成 SBN 同步  
  // Synchronisation SBN complète.

  MPI_Barrier(MPI_COMM_WORLD);
  // MPI 全局同步  
  // Synchronisation globale MPI.
#endif

#if USE_MPI
  double start = MPI_Wtime();
  // MPI 下计时开始  
  // Début du chronométrage MPI.
#else
  timeval start;
  gettimeofday(&start, NULL);
  // 非 MPI 下使用 wall clock 计时  
  // Chronométrage wall-clock pour mode non-MPI.
#endif

  while ((locDom->time() < locDom->stoptime()) &&
         (locDom->cycle() < opts.its)) {
    // 主循环：时间未结束且未达到迭代上限  
    // Boucle principale : temps non terminé et limite d’itération non atteinte.

    TimeIncrement(*locDom);
    // 更新时间步  
    // Mise à jour du pas de temps.

    LagrangeLeapFrog(*locDom);
    // 主 Lagrange 跳步算法  
    // Algorithme Leapfrog principal.

    if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
      printf("cycle = %d, time = %e, dt=%e\n", locDom->cycle(),
             double(locDom->time()), double(locDom->deltatime()));
      // 输出当前进度  
      // Affiche la progression courante.
    }
  }

  double elapsed_time;
#if USE_MPI
  elapsed_time = MPI_Wtime() - start;
  // MPI 模式：时间差  
  // Mode MPI : différence de temps.
#else
  timeval end;
  gettimeofday(&end, NULL);
  elapsed_time = (double)(end.tv_sec - start.tv_sec) +
                 ((double)(end.tv_usec - start.tv_usec)) / 1000000;
  // 非 MPI 模式：wall clock 计算运行时间  
  // Mode non-MPI : temps écoulé via wall clock.
#endif

  double elapsed_timeG;
#if USE_MPI
  MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  // 收集最大运行时间（确保正确）  
  // Réduction MPI pour obtenir le temps maximal.
#else
  elapsed_timeG = elapsed_time;
  // 单进程直接赋值  
  // Valeur directe pour mode mono-processus.
#endif

  if (opts.viz) {
    DumpToVisit(*locDom, opts.numFiles, myRank, numRanks);
    // 输出 VisIt 文件  
    // Génère les fichiers VisIt.
  }

  if ((myRank == 0) && (opts.quiet == 0)) {
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
    // 验证结果并输出最终信息  
    // Vérifie les résultats et écrit la sortie finale.
  }

  Kokkos::finalize();
  // 关闭 Kokkos  
  // Finalisation de Kokkos.

#if USE_MPI
  MPI_Finalize();
  // 结束 MPI  
  // Finalisation MPI.
#endif

  return 0;
  // 正常结束  
  // Fin normale du programme.
}
