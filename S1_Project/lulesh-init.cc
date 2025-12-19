#include <cmath>          // math operations (preferred)
#include <cstdio>         // printf
#include <cstdlib>        // rand, exit
#include <cstring>        // memset, memcpy
#include <limits.h>       // INT_MIN / INT_MAX
#include <algorithm>      // std::min/max
#include <Kokkos_Random.hpp>

#include <Kokkos_Core.hpp>    // Kokkos 核心头文件 

#if USE_MPI
#include <mpi.h>
#endif

#if _OPENMP
#include <omp.h>
#endif

#include "lulesh.h"

/* 
 * Constructeur du domaine — version Kokkos 5.0
 * Domain 构造函数 —— Kokkos 5.0 版本
 */
Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, Int_t tp, Int_t nr,
               Int_t balance, Int_t cost)
  :
    /* Constantes physiques / 物理常数初始化 */
    m_e_cut(Real_t(1.0e-7)),
    m_p_cut(Real_t(1.0e-7)),
    m_q_cut(Real_t(1.0e-7)),
    m_v_cut(Real_t(1.0e-10)),
    m_u_cut(Real_t(1.0e-7)),
    m_hgcoef(Real_t(3.0)),
    m_ss4o3(Real_t(4.0)/Real_t(3.0)),
    m_qstop(Real_t(1.0e+12)),
    m_monoq_max_slope(Real_t(1.0)),
    m_monoq_limiter_mult(Real_t(2.0)),
    m_qlc_monoq(Real_t(0.5)),
    m_qqc_monoq(Real_t(2.0)/Real_t(3.0)),
    m_qqc(Real_t(2.0)),
    m_eosvmax(Real_t(1.0e+9)),
    m_eosvmin(Real_t(1.0e-9)),
    m_pmin(Real_t(0.0)),
    m_emin(Real_t(-1.0e+15)),
    m_dvovmax(Real_t(0.1)),
    m_refdens(Real_t(1.0)),
    m_numReg(nr),
    m_cost(cost),
    m_numRanks(numRanks)
{
    /* 
     * Dimensions globales / 全局网格尺寸
     */
    m_tp       = tp;
    m_colLoc   = colLoc;
    m_rowLoc   = rowLoc;
    m_planeLoc = planeLoc;

    Index_t edgeElems = nx;
    Index_t edgeNodes = nx + 1;

    m_sizeX = edgeElems;
    m_sizeY = edgeElems;
    m_sizeZ = edgeElems;
    m_numElem = edgeElems * edgeElems * edgeElems;
    m_numNode = edgeNodes * edgeNodes * edgeNodes;

    /* 
     * Allocation des champs nodaux et élémentaires
     * 分配节点场与单元场 (Kokkos::View)
     */
    AllocateElemPersistent(m_numElem);
    AllocateNodePersistent(m_numNode);

    /* 
     * Construction du maillage initial
     * 构建初始规则网格
     */
    BuildMesh(nx, edgeNodes, edgeElems);

    /* 
     * Initialisation aléatoire（用于 region sets）
     * 使用 Kokkos random pool 作为随机源
     */
    Kokkos::Random_XorShift64_Pool<> randPool(12345);

    /*
     * Initialisation des champs élémentaires
     * 单元物理量初始化为 0
     */
    {
        using ES = Kokkos::DefaultExecutionSpace;

        Kokkos::parallel_for(
            "InitElemFields",
            Kokkos::RangePolicy<ES>(0, m_numElem),
            KOKKOS_LAMBDA(Index_t i) {
                // énergie / 内能
                // pression / 压力
                // viscosité artificielle / 人工粘性
                // vitesse du son / 声速
                e(i)  = Real_t(0.0);
                p(i)  = Real_t(0.0);
                q(i)  = Real_t(0.0);
                ss(i) = Real_t(0.0);

                // v initialise à 1.0 / 相对体积初始化为 1.0
                v(i) = Real_t(1.0);
            }
        );
    }

    /*
     * Initialisation des champs nodaux
     * 节点速度 / 加速度 / 节点质量初始化为 0
     */
    {
        using ES = Kokkos::DefaultExecutionSpace;

        Kokkos::parallel_for(
            "InitNodeFields",
            Kokkos::RangePolicy<ES>(0, m_numNode),
            KOKKOS_LAMBDA(Index_t i) {
                xd(i) = Real_t(0.0);
                yd(i) = Real_t(0.0);
                zd(i) = Real_t(0.0);

                xdd(i) = Real_t(0.0);
                ydd(i) = Real_t(0.0);
                zdd(i) = Real_t(0.0);

                nodalMass(i) = Real_t(0.0);
            }
        );
    }

    /*
     * Buffers de communication MPI（CPU-only）
     * MPI 缓冲区（仅保持接口，不做操作）
     */
    SetupCommBuffers(edgeNodes);

    /*
     * Création des régions物
     * 随机区域划分（使用 random pool, 主机完成）
     */
    // CreateRegionIndexSets_Kokkos(nr, balance, randPool);
    CreateRegionIndexSets(nr, balance);

    /*
     * Plans de symétrie / 对称平面
     */
    SetupSymmetryPlanes(edgeNodes);

    /*
     * Connectivités élémentaires / 单元连接关系
     */
    SetupElementConnectivities(edgeElems);

    /*
     * Conditions limites / 边界条件
     */
    SetupBoundaryConditions(edgeElems);
    SetupInitialVolumesAndMasses();
    SetupThreadSupportStructures();

    // --- Sedov blast: initial energy + initial timestep (match reference LULESH)
    {
      const Real_t ebase = Real_t(3.948746e+7);
      const Real_t scale = Real_t(nx * m_tp) / Real_t(45.0);
      const Real_t einit = ebase * scale * scale * scale;

      // deposit initial energy in origin element
      if ((m_rowLoc + m_colLoc + m_planeLoc) == 0) {
        e(0) = einit;
      }
      // set initial dt from element-0 volume and deposited energy
      deltatime() = (Real_t(0.5) * CBRT(volo(0))) / SQRT(Real_t(2.0) * einit);
    }



    /*
     * Valeurs par défaut（时间推进）
     * 初值设定（时间推进参数）
     */
    dtfixed()        = Real_t(-1.0e-6);
    stoptime()       = Real_t(1.0e-2);
    deltatimemultlb() = Real_t(1.1);
    deltatimemultub() = Real_t(1.2);
    dtcourant()      = Real_t(1.0e+20);
    dthydro()        = Real_t(1.0e+20);
    dtmax()          = Real_t(1.0e-2);
    time()           = Real_t(0.0);
    cycle()          = Int_t(0);
}

// -----------------------------------------------------------------------------
// Missing Domain methods (Kokkos 5.0 port, CPU/Serial-friendly)
// 缺失的 Domain 成员函数补齐（Kokkos 5.0 / CPU 串行版）
// -----------------------------------------------------------------------------

// Out-of-line destructor definition (fixes linker undefined symbol)
Domain::~Domain() = default;

// Serial build: keep interface, no MPI comm buffers.
void Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // Flags used by SetupBoundaryConditions(): 0 => boundary (no neighbor), 1 => has neighbor
  m_rowMin   = (m_rowLoc   == 0      ) ? 0 : 1;
  m_rowMax   = (m_rowLoc   == m_tp-1 ) ? 0 : 1;
  m_colMin   = (m_colLoc   == 0      ) ? 0 : 1;
  m_colMax   = (m_colLoc   == m_tp-1 ) ? 0 : 1;
  m_planeMin = (m_planeLoc == 0      ) ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1 ) ? 0 : 1;

  const Index_t planeNodes = Index_t(edgeNodes) * Index_t(edgeNodes);

  // Allocate symmetry-plane node lists only on the global min faces
  // (and reset to empty views otherwise so symmXempty() works correctly)
  if (m_colLoc == 0)   m_symmX = Kokkos::View<Index_t*>("symmX", planeNodes);
  else                 m_symmX = Kokkos::View<Index_t*>();

  if (m_rowLoc == 0)   m_symmY = Kokkos::View<Index_t*>("symmY", planeNodes);
  else                 m_symmY = Kokkos::View<Index_t*>();

  if (m_planeLoc == 0) m_symmZ = Kokkos::View<Index_t*>("symmZ", planeNodes);
  else                 m_symmZ = Kokkos::View<Index_t*>();

#ifdef USE_MPI
  // If MPI buffers are still needed elsewhere, restore official allocations here.
#endif
}

// Optional: keep symbol available; can be implemented later if needed.
void Domain::SetupThreadSupportStructures()
{
  // no-op for now (serial)
}

/*
 * BuildMesh — build regular mesh coordinates + element connectivity.
 * 中文：构建规则立方网格的节点坐标(x,y,z)与单元8节点连接(nodelist)
 *
 * 说明：这里用 Domain 的现有访问接口 x()/y()/z() 和 nodelist(elem)
 *       不假设你内部 View 的维度/名字，避免与你当前改造冲突。
 */
void Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  const Index_t meshEdgeElems =
      static_cast<Index_t>(m_tp) * static_cast<Index_t>(nx);

  // -----------------------
  // 1) Node coordinates
  // -----------------------
  Index_t nidx = 0;

  Real_t tz = Real_t(1.125) *
              Real_t(static_cast<Index_t>(m_planeLoc) * nx) /
              Real_t(meshEdgeElems);

  for (Index_t plane = 0; plane < static_cast<Index_t>(edgeNodes); ++plane) {

    Real_t ty = Real_t(1.125) *
                Real_t(static_cast<Index_t>(m_rowLoc) * nx) /
                Real_t(meshEdgeElems);

    for (Index_t row = 0; row < static_cast<Index_t>(edgeNodes); ++row) {

      Real_t tx = Real_t(1.125) *
                  Real_t(static_cast<Index_t>(m_colLoc) * nx) /
                  Real_t(meshEdgeElems);

      for (Index_t col = 0; col < static_cast<Index_t>(edgeNodes); ++col) {
        x(nidx) = tx;
        y(nidx) = ty;
        z(nidx) = tz;
        ++nidx;

        tx = Real_t(1.125) *
             Real_t(static_cast<Index_t>(m_colLoc) * nx + col + 1) /
             Real_t(meshEdgeElems);
      }

      ty = Real_t(1.125) *
           Real_t(static_cast<Index_t>(m_rowLoc) * nx + row + 1) /
           Real_t(meshEdgeElems);
    }

    tz = Real_t(1.125) *
         Real_t(static_cast<Index_t>(m_planeLoc) * nx + plane + 1) /
         Real_t(meshEdgeElems);
  }

  // -----------------------
  // 2) Element connectivity (8 nodes per hex)
  // -----------------------
  Index_t zidx = 0;
  nidx = 0;

  for (Index_t plane = 0; plane < static_cast<Index_t>(edgeElems); ++plane) {
    for (Index_t row = 0; row < static_cast<Index_t>(edgeElems); ++row) {
      for (Index_t col = 0; col < static_cast<Index_t>(edgeElems); ++col) {

        Index_t* nl = nodelist(zidx);

        const Index_t base        = nidx;
        const Index_t stride      = static_cast<Index_t>(edgeNodes);
        const Index_t planeStride = stride * stride;

        nl[0] = base;
        nl[1] = base + 1;
        nl[2] = base + stride + 1;
        nl[3] = base + stride;
        nl[4] = base + planeStride;
        nl[5] = base + planeStride + 1;
        nl[6] = base + planeStride + stride + 1;
        nl[7] = base + planeStride + stride;

        ++zidx;
        ++nidx;
      }
      ++nidx; // skip last node in row
    }
    nidx += static_cast<Index_t>(edgeNodes); // skip last row in plane
  }
}


/*--------------------------------------------------------------*
 |   Partie A.2 — Initialisation volumique et énergie initiale  |
 |   A.2 部分 — 单元体积、质量与初始能量初始化（Kokkos 版本）   |
 *--------------------------------------------------------------*/


    /* 
     * Calcul des volumes initiaux volo(i) et des masses nodales
     * 并行计算初始体积 volo(i) 与 nodalMass（每个单元贡献 1/8）
     *
     * 注意：需访问单元的 8 个节点坐标，因此使用 parallel_for
     */
    
void Domain::SetupInitialVolumesAndMasses()
{

    using ES = Kokkos::DefaultExecutionSpace;

    Kokkos::parallel_for(
        "ComputeInitialVolumesAndMasses",
        Kokkos::RangePolicy<ES>(0, m_numElem),
        KOKKOS_LAMBDA(const Index_t i)
    {
        Real_t x_local[8], y_local[8], z_local[8];

        /* 
         * Charger les coordonnées nodales pour l’élément i
         * 加载单元 i 对应的 8 个节点坐标
         */
        Index_t* elemToNode = nodelist(i);
        for (Index_t lnode = 0; lnode < 8; ++lnode) {
            Index_t g = elemToNode[lnode];
            x_local[lnode] = x(g);
            y_local[lnode] = y(g);
            z_local[lnode] = z(g);
        }

        /* 
         * Volume initial volo(i)
         * 初始单元体积
         */
        Real_t volume = CalcElemVolume(x_local, y_local, z_local);
        volo(i) = volume;

        /* 
         * La masse élémentaire = volume (densité initiale = 1)
         * 单元质量 = 初始体积（密度 = 1）
         */
        elemMass(i) = volume;

        /* 
         * Répartition de masse : chaque nœud reçoit 1/8 de la masse
         * 将单元质量均匀分配给 8 个节点
         */
        Real_t nodalShare = volume / Real_t(8.0);

        for (Index_t j = 0; j < 8; ++j) {
            Index_t g = elemToNode[j];
            Kokkos::atomic_add(&nodalMass(g), nodalShare);
        }
    });

}
/*--------------------------------------------------------------------*
 |   Dépôt d’énergie initiale (Sedov blast)                            |
 |   初始能量注入（Sedov 爆炸问题的初值设定）                           |
 *--------------------------------------------------------------------*/



void Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
  // 中文：初始化随机池（用于区域分布）
  // 法语：Initialiser le pool aléatoire (pour la distribution des régions)
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Kokkos::Random_XorShift64_Pool<ExecSpace> randPool(12345);


  // 中文：设置区域数量
  // 法语：Définir le nombre total de régions
  numReg() = nr;

  // 中文：为区域尺寸数组分配空间（大小 = nr）
  // 法语：Allouer l’espace pour la taille des régions (taille = nr)
  m_regElemSize = Kokkos::View<Index_t*>("regElemSize", nr);

  // 中文：分配 regNumList（每个元素属于哪个区域）
  // 法语：Allouer regNumList (chaque élément → numéro de région)
  m_regNumList = Kokkos::View<Index_t*>("regNumList", numElem());

  // 中文：regElemlist 是二维 View（每个区域一个列表）
  // 法语：regElemlist est un View 2D (une liste par région)
  // 第一阶段还不知道每个区域大小，稍后 allocate。
  m_regElemlist = Kokkos::View<Index_t**>("regElemlist", nr, numElem());

  // ------------------------------
  // Host mirrors（因为区域生成是严格串行算法）
  // ------------------------------
  auto h_regElemSize = Kokkos::create_mirror_view(m_regElemSize);
  auto h_regNumList  = Kokkos::create_mirror_view(m_regNumList);

  // 中文：用于权重概率分布
  // 法语：Distribution de probabilité basée sur les poids
  std::vector<Int_t> regBinEnd(nr);

  Int_t nextIndex = 0;
  Int_t lastReg = -1;

  // ------------------------------
  // 情况 1：只存在一个区域
  // ------------------------------
  if (nr == 1) {
    const Index_t ne = numElem();

    m_regNumList = Kokkos::View<Index_t*>("regNumList", ne);
    Kokkos::deep_copy(m_regNumList, Index_t(0));

    m_regElemSize = Kokkos::View<Index_t*>("regElemSize", 1);
    auto regSize_h = Kokkos::create_mirror_view(m_regElemSize);
    regSize_h(0) = ne;
    Kokkos::deep_copy(m_regElemSize, regSize_h);

    m_regElemlist = Kokkos::View<Index_t**>("regElemlist", 1, ne);
    auto regList_h = Kokkos::create_mirror_view(m_regElemlist);
    for (Index_t i = 0; i < ne; ++i) regList_h(0, i) = i;
    Kokkos::deep_copy(m_regElemlist, regList_h);

    return;
  }

  // ------------------------------
  // 情况 2：多个区域，随机分配
  // ------------------------------

  // 中文：计算区域权重分布  Σ (i+1)^balance
  // 法语：Calculer les poids des régions  Σ (i+1)^balance
  Int_t costDen = 0;
  for (Int_t i=0; i<nr; ++i) {
    costDen += std::pow(i+1, balance);
    regBinEnd[i] = costDen;
    h_regElemSize(i) = 0;
  }

  // 中文：从 RandomPool 获取 RNG 实例
  // 法语：Obtenir un générateur aléatoire depuis le RandomPool
  auto rand_gen = randPool.get_state();

  while (nextIndex < numElem()) {

    // -------- 区域选择：按权重 --------
    Int_t rnum = static_cast<Int_t>(rand_gen.urand() * costDen);
    Int_t idx = 0;
    while (rnum >= regBinEnd[idx]) idx++;

    // 中文：旋转区域编号，使不同 rank 分布不同（目前 rank=0）
    // 法语：Rotation du numéro de région (rank=0 dans ce cas)
    Int_t region = idx + 1;

    // 中文：避免连续两次选同一区域
    // 法语：Éviter de choisir deux fois de suite la même région
    while (region == lastReg) {
      static_cast<Int_t>(rand_gen.urand() * costDen);
      idx = 0;
      while (rnum >= regBinEnd[idx]) idx++;
      region = idx + 1;
    }

    // -------- 计算 binSize（决定一次分配多少元素） --------
    Int_t binSize = static_cast<Int_t>(rand_gen.urand() * 1000);
    Index_t elements;

    if (binSize < 773)        elements = static_cast<Int_t>(rand_gen.urand() * 15) + 1;
    else if (binSize < 937)   elements = static_cast<Int_t>(rand_gen.urand() * 16) + 16;
    else if (binSize < 970)   elements = static_cast<Int_t>(rand_gen.urand() * 32) + 32;
    else if (binSize < 974)   elements = static_cast<Int_t>(rand_gen.urand() * 64) + 64;
    else if (binSize < 978)   elements = static_cast<Int_t>(rand_gen.urand() * 128) + 128;
    else if (binSize < 981)   elements = static_cast<Int_t>(rand_gen.urand() * 256) + 256;
    else                      elements = static_cast<Int_t>(rand_gen.urand() * 1537) + 512;

    Index_t runto = nextIndex + elements;

    // -------- 将元素分配到区域 --------
    while (nextIndex < runto && nextIndex < numElem()) {
      h_regNumList(nextIndex) = region;
      nextIndex++;
      h_regElemSize(region-1)++;
    }

    lastReg = region;
  }

  // 归还 RNG
  randPool.free_state(rand_gen);

  // 同步回 device
  Kokkos::deep_copy(m_regElemSize, h_regElemSize);
  Kokkos::deep_copy(m_regNumList,  h_regNumList);

  // -----------------------------
  // 建立 regElemlist（二次扫描）
  // -----------------------------
  auto h_regElemlist = Kokkos::create_mirror_view(m_regElemlist);

  // 清零计数器
  std::vector<Index_t> offset(nr, 0);

  for (Index_t elem = 0; elem < numElem(); ++elem) {
    Index_t r = h_regNumList(elem) - 1;
    Index_t pos = offset[r]++;
    h_regElemlist(r, pos) = elem;
  }

  // 最终同步到 device
  Kokkos::deep_copy(m_regElemlist, h_regElemlist);
}

void Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
  const Index_t planeNodes = Index_t(edgeNodes) * Index_t(edgeNodes);

  // Node index mapping: n(i,j,k) = i + edgeNodes*(j + edgeNodes*k)
  // We flatten (a,b) in [0..edgeNodes-1]^2 as nidx = a*edgeNodes + b.
  // Then:
  //  X=0 plane: (i=0, j=a, k=b)
  //  Y=0 plane: (i=a, j=0, k=b)
  //  Z=0 plane: (i=a, j=b, k=0)

  if (!symmXempty()) {
    if (Index_t(m_symmX.extent(0)) != planeNodes) m_symmX = Kokkos::View<Index_t*>("symmX", planeNodes);
    auto hx = Kokkos::create_mirror_view(m_symmX);
    for (Index_t a = 0; a < edgeNodes; ++a) {
      for (Index_t b = 0; b < edgeNodes; ++b) {
        const Index_t nidx = a * edgeNodes + b;
        hx(nidx) = Index_t(0) + Index_t(edgeNodes) * (a + Index_t(edgeNodes) * b);
      }
    }
    Kokkos::deep_copy(m_symmX, hx);
  }

  if (!symmYempty()) {
    if (Index_t(m_symmY.extent(0)) != planeNodes) m_symmY = Kokkos::View<Index_t*>("symmY", planeNodes);
    auto hy = Kokkos::create_mirror_view(m_symmY);
    for (Index_t a = 0; a < edgeNodes; ++a) {
      for (Index_t b = 0; b < edgeNodes; ++b) {
        const Index_t nidx = a * edgeNodes + b;
        hy(nidx) = a + Index_t(edgeNodes) * (Index_t(0) + Index_t(edgeNodes) * b);
      }
    }
    Kokkos::deep_copy(m_symmY, hy);
  }

  if (!symmZempty()) {
    if (Index_t(m_symmZ.extent(0)) != planeNodes) m_symmZ = Kokkos::View<Index_t*>("symmZ", planeNodes);
    auto hz = Kokkos::create_mirror_view(m_symmZ);
    for (Index_t a = 0; a < edgeNodes; ++a) {
      for (Index_t b = 0; b < edgeNodes; ++b) {
        const Index_t nidx = a * edgeNodes + b;
        hz(nidx) = a + Index_t(edgeNodes) * b;
      }
    }
    Kokkos::deep_copy(m_symmZ, hz);
  }
}

void Domain::SetupElementConnectivities(Int_t edgeElems)
{
  /* 
     Français : Préparer la connectivité élément → voisin dans les trois directions (ξ, η, ζ).
     中文：初始化单元在三方向（ξ, η, ζ）上的邻接关系（前向与后向）。
  */

  // ------------------------------------------------------------
  // Direction ξ (colonne) : lxim / lxip
  // ------------------------------------------------------------
  // Français : Pour ξ-, le premier élément n’a pas de voisin → s’auto-référence.
  // 中文：ξ 方向负侧的第一个单元没有邻居 → 指向自己。
  lxim(0) = 0;

  for (Index_t i = 1; i < numElem(); ++i) {
    // Français : voisin ξ- = élément précédent
    // 中文：ξ 负方向邻居 = 前一个单元
    lxim(i) = i - 1;

    // Français : voisin ξ+ du précédent = cet élément
    // 中文：ξ 正方向邻居（前一个单元的正向）= 当前单元
    lxip(i - 1) = i;
  }

  // Français : Le dernier élément n’a pas de voisin ξ+ → auto-référence.
  // 中文：最后一个单元在 ξ+ 方向没有邻居 → 指向自己。
  lxip(numElem() - 1) = numElem() - 1;

  // ------------------------------------------------------------
  // Direction η (ligne) : letam / letap
  // ------------------------------------------------------------
  for (Index_t i = 0; i < edgeElems; ++i) {
    // Français : bord η- → pas de voisin, auto-référence
    // 中文：η 最小边界 → 无邻居，指向自己
    letam(i) = i;

    // Français : bord η+ → dernier plan, auto-référence
    // 中文：η 最大边界 → 最后一个平面，指向自己
    letap(numElem() - edgeElems + i) = numElem() - edgeElems + i;
  }

  for (Index_t i = edgeElems; i < numElem(); ++i) {
    // Français : voisin η- = un plan plus bas
    // 中文：η 方向负邻居 = 往下移动一个平面
    letam(i) = i - edgeElems;

    // Français : voisin η+ du plan inférieur = cet élément
    // 中文：η 正方向邻居（下一个平面）= 当前单元
    letap(i - edgeElems) = i;
  }

  // ------------------------------------------------------------
  // Direction ζ (plan) : lzetam / lzetap
  // ------------------------------------------------------------
  for (Index_t i = 0; i < edgeElems * edgeElems; ++i) {
    // Français : bord ζ- → auto-référence
    // 中文：ζ 最小面 → 指向自己
    lzetam(i) = i;

    // Français : bord ζ+ du dernier plan → auto-référence
    // 中文：ζ 最大面（最后一个大平面）→ 指向自己
    lzetap(numElem() - edgeElems * edgeElems + i) =
        numElem() - edgeElems * edgeElems + i;
  }

  for (Index_t i = edgeElems * edgeElems; i < numElem(); ++i) {
    // Français : voisin ζ- = un bloc de edgeElems² éléments plus bas
    // 中文：ζ 负方向邻居 = 往下移动 edgeElems² 个单元（跨一个大平面）
    lzetam(i) = i - edgeElems * edgeElems;

    // Français : voisin ζ+ du bloc inférieur = cet élément
    // 中文：ζ 正方向邻居（上一个大平面）= 当前单元
    lzetap(i - edgeElems * edgeElems) = i;
  }
}

void Domain::SetupBoundaryConditions(Int_t edgeElems)
{
  /*
     Français : Initialiser les conditions aux limites pour chaque élément du maillage.
                Inclut les plans de symétrie, surfaces libres, et références aux fantômes.
     中文：为网格中的每个单元初始化边界条件，包括对称面、自由面、以及幽灵单元的引用。
  */

  Index_t ghostIdx[6];

  // ------------------------------------------------------------
  // 1) Initialiser elemBC : aucune condition au début
  // ------------------------------------------------------------
  // Français : aucune condition appliquée au départ
  // 中文：初始所有单元边界条件设为 0
  for (Index_t i = 0; i < numElem(); ++i) {
    elemBC(i) = Int_t(0);
  }

  // ------------------------------------------------------------
  // 2) Initialiser ghostIdx[] avec INT_MIN (invalide)
  // ------------------------------------------------------------
  // Français : valeurs invalides par défaut (aucun fantôme)
  // 中文：默认无幽灵单元，用 INT_MIN 标记
  for (Index_t i = 0; i < 6; ++i) {
    ghostIdx[i] = INT_MIN;
  }

  // ------------------------------------------------------------
  // 3) Calculer offsets pour les ghost cells
  // ------------------------------------------------------------
  // Français : position de départ pour stocker les fantômes
  // 中文：确定幽灵单元在扩展域中的偏移起始位置
  Int_t pidx = numElem();

  if (m_planeMin != 0) {
    ghostIdx[0] = pidx;
    pidx += sizeX() * sizeY();
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx;
    pidx += sizeX() * sizeY();
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx;
    pidx += sizeX() * sizeZ();
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx;
    pidx += sizeX() * sizeZ();
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx;
    pidx += sizeY() * sizeZ();
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx;
  }

  // ------------------------------------------------------------
  // 4) Appliquer conditions aux limites pour chaque élément
  // ------------------------------------------------------------
  // Français : parcourir chaque couche, ligne, colonne
  // 中文：逐层逐行逐列初始化边界条件
  for (Index_t i = 0; i < edgeElems; ++i) {
    Index_t planeInc = i * edgeElems * edgeElems;
    Index_t rowInc   = i * edgeElems;

    for (Index_t j = 0; j < edgeElems; ++j) {

      // --------------------------------------------------------
      // ZETA_M : direction ζ- (面向最小 plane)
      // --------------------------------------------------------
      if (m_planeLoc == 0) {
        // Français : symétrie au plan ζ-
        // 中文：ζ− 方向为对称面
        elemBC(rowInc + j) |= ZETA_M_SYMM;
      } else {
        elemBC(rowInc + j) |= ZETA_M_COMM;
        lzetam(rowInc + j) = ghostIdx[0] + rowInc + j;
      }

      // --------------------------------------------------------
      // ZETA_P : direction ζ+ (最大 plane)
      // --------------------------------------------------------
      Index_t zpIdx = rowInc + j + numElem() - edgeElems * edgeElems;
      if (m_planeLoc == m_tp - 1) {
        elemBC(zpIdx) |= ZETA_P_FREE;  // surface libre
      } else {
        elemBC(zpIdx) |= ZETA_P_COMM;
        lzetap(zpIdx) = ghostIdx[1] + rowInc + j;
      }

      // --------------------------------------------------------
      // ETA_M : direction η- (最小 row)
      // --------------------------------------------------------
      if (m_rowLoc == 0) {
        elemBC(planeInc + j) |= ETA_M_SYMM;
      } else {
        elemBC(planeInc + j) |= ETA_M_COMM;
        letam(planeInc + j) = ghostIdx[2] + rowInc + j;
      }

      // --------------------------------------------------------
      // ETA_P : direction η+ (最大 row)
      // --------------------------------------------------------
      Index_t epIdx = planeInc + j + edgeElems * edgeElems - edgeElems;
      if (m_rowLoc == m_tp - 1) {
        elemBC(epIdx) |= ETA_P_FREE;
      } else {
        elemBC(epIdx) |= ETA_P_COMM;
        letap(epIdx) = ghostIdx[3] + rowInc + j;
      }

      // --------------------------------------------------------
      // XI_M : direction ξ- (最小 col)
      // --------------------------------------------------------
      if (m_colLoc == 0) {
        elemBC(planeInc + j * edgeElems) |= XI_M_SYMM;
      } else {
        elemBC(planeInc + j * edgeElems) |= XI_M_COMM;
        lxim(planeInc + j * edgeElems) = ghostIdx[4] + rowInc + j;
      }

      // --------------------------------------------------------
      // XI_P : direction ξ+ (最大 col)
      // --------------------------------------------------------
      Index_t xpIdx = planeInc + j * edgeElems + edgeElems - 1;
      if (m_colLoc == m_tp - 1) {
        elemBC(xpIdx) |= XI_P_FREE;
      } else {
        elemBC(xpIdx) |= XI_P_COMM;
        lxip(xpIdx) = ghostIdx[5] + rowInc + j;
      }
    }
  }
}

void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
  /*
     Français : Décomposer le domaine global en un maillage 3D régulier
                selon un agencement cubique des sous-domaines.
     中文：将整体模拟域按立方体方式划分为多个子域（用于多核或多 MPI 进程）。
  */

  Int_t testProcs;
  Int_t dx, dy, dz;
  Int_t myDom;

  // ------------------------------------------------------------
  // Vérifier que le nombre de domaines est un cube parfait
  // ------------------------------------------------------------
  // Français : LULESH exige 1, 8, 27, ...
  // 中文：LULESH 要求进程数量必须是整数立方（1、8、27…）
  testProcs = Int_t(cbrt(Real_t(numRanks)) + 0.5);

  if (testProcs * testProcs * testProcs != numRanks) {
    printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n");
    exit(-1);   // MPI absent → utiliser exit()
  }

  // ------------------------------------------------------------
  // Vérifier que le type Real_t est supporté
  // ------------------------------------------------------------
  // Français : seules float/double sont autorisées
  // 中文：只允许 float/double 两种浮点类型
  if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
    printf("MPI operations only support float and double right now...\n");
    exit(-1);
  }

  // ------------------------------------------------------------
  // Vérifier la taille des buffers pour la communication fantôme
  // ------------------------------------------------------------
  // Français : limitation structurelle de LULESH original
  // 中文：保证用于幽灵单元交换的内部缓冲区大小正确
  if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
    printf("corner element comm buffers too small.  Fix code.\n");
    exit(-1);
  }

  // ------------------------------------------------------------
  // Distribution cubique des sous-domaines
  // ------------------------------------------------------------
  dx = testProcs;
  dy = testProcs;
  dz = testProcs;

  if (dx * dy * dz != numRanks) {
    printf("error -- must have as many domains as procs\n");
    exit(-1);
  }

  // ------------------------------------------------------------
  // Répartition des sous-domaines entre les rangs
  // ------------------------------------------------------------
  // Français : distribution équilibrée même si numRanks ne divise pas parfaitement
  // 中文：在理论上允许不均匀划分，但这里确保均匀分布
  Int_t remainder = dx * dy * dz % numRanks;

  if (myRank < remainder) {
    myDom = myRank * (1 + (dx * dy * dz / numRanks));
  } else {
    myDom = remainder * (1 + (dx * dy * dz / numRanks)) +
            (myRank - remainder) * (dx * dy * dz / numRanks);
  }

  // ------------------------------------------------------------
  // Convertir myDom en coordonnées 3D dans la grille
  // ------------------------------------------------------------
  // Français : conversion index → (col, row, plane)
  // 中文：将一维编号转映射为三维坐标
  *col   = myDom % dx;
  *row   = (myDom / dx) % dy;
  *plane = myDom / (dx * dy);
  *side  = testProcs;  // 每边长度

  return;
}
