#include <math.h>
#if USE_MPI
# include <mpi.h>
#endif
#if _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include "lulesh.h"

Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
    :
      m_e_cut(Real_t(1.0e-7)),   // 截断能量阈值
                                 // seuil de coupure pour l'énergie
      m_p_cut(Real_t(1.0e-7)),   // 截断压力阈值
                                 // seuil de coupure pour la pression
      m_q_cut(Real_t(1.0e-7)),   // 截断人工黏性阈值
                                 // seuil pour la viscosité artificielle
      m_v_cut(Real_t(1.0e-10)),  // 体积截断阈值
                                 // seuil de coupure du volume
      m_u_cut(Real_t(1.0e-7)),   // 速度截断阈值
                                 // seuil de coupure de la vitesse
      m_hgcoef(Real_t(3.0)),     // hourglass 系数
                                 // coefficient anti-hourglass
      m_ss4o3(Real_t(4.0)/Real_t(3.0)), // 声速比例系数
                                       // coefficient acoustique (4/3)
      m_qstop(Real_t(1.0e+12)),  // Q 上限
                                 // limite supérieure du Q
      m_monoq_max_slope(Real_t(1.0)),    // 单调 Q 最大斜率
                                         // pente max pour Monotonic Q
      m_monoq_limiter_mult(Real_t(2.0)), // limiter 系数
                                         // multiplicateur pour le limiteur
      m_qlc_monoq(Real_t(0.5)),          // 线性黏性参数
                                         // viscosité linéaire
      m_qqc_monoq(Real_t(2.0)/Real_t(3.0)), // 二次黏性参数
                                            // viscosité quadratique
      m_qqc(Real_t(2.0)),      // Q 计算常数
                               // constante pour le calcul de Q
      m_eosvmax(Real_t(1.0e+9)), // EOS 最大体积
                                 // volume maximal pour l’EOS
      m_eosvmin(Real_t(1.0e-9)), // EOS 最小体积
                                 // volume minimal pour l’EOS
      m_pmin(Real_t(0.0)),       // 压力下限
                                 // pression minimale
      m_emin(Real_t(-1.0e+15)),  // 能量下限
                                 // énergie minimale
      m_dvovmax(Real_t(0.1)),    // 最大体积变化率
                                 // taux max de changement de volume
      m_refdens(Real_t(1.0))     // 初始密度
                                 // densité de référence
{
    Index_t edgeElems = nx;             // 每边单元数
                                        // nombre d’éléments par côté
    Index_t edgeNodes = edgeElems + 1;  // 每边节点数
                                        // nombre de nœuds par côté

    this->cost() = cost; // 设置 region 成本
                         // définie le coût de région

    m_tp       = tp;      // 线程/进程划分参数
                          // paramètre de décomposition
    m_numRanks = numRanks; // MPI 总进程数
                           // nombre total de rangs MPI

    m_colLoc   = colLoc;   // 本 rank 在 X 方向的位置
                           // position du rang en X
    m_rowLoc   = rowLoc;   // 本 rank 在 Y 方向的位置
                           // position du rang en Y
    m_planeLoc = planeLoc; // 本 rank 在 Z 方向的位置
                           // position du rang en Z
    
    m_sizeX = edgeElems;   // 本子域尺寸 X
                           // taille du sous-domaine en X
    m_sizeY = edgeElems;   // 本子域尺寸 Y
                           // taille du sous-domaine en Y
    m_sizeZ = edgeElems;   // 本子域尺寸 Z
                           // taille du sous-domaine en Z

    m_numElem = edgeElems * edgeElems * edgeElems;  
    // 本子域包含的单元数量
    // nombre total d’éléments dans ce sous-domaine

    m_numNode = edgeNodes * edgeNodes * edgeNodes;
    // 本子域包含的节点数量
    // nombre total de nœuds dans ce sous-domaine

    m_regNumList = new Index_t[numElem()];
    // region 编号数组（遗留：此处保持原始 new，不进入 Kokkos）
    // tableau des régions (hérité : reste en new ici)

    AllocateElemPersistent(numElem());
    // 分配单元相关的 persistent View
    // allocation des View persistants pour les éléments

    AllocateNodePersistent(numNode());
    // 分配节点相关的 persistent View
    // allocation des View persistants pour les nœuds

    SetupCommBuffers(edgeNodes);
    // MPI 边界数据缓冲区初始化
    // initialisation des buffers de communication

        // 单元场初始化：e, p, q, ss 全为 0
    // Initialisation des champs élémentaires : e, p, q, ss à zéro
    Kokkos::parallel_for("InitElemFields", numElem(), KOKKOS_LAMBDA(const Index_t i) {
        e(i)  = Real_t(0.0);
        p(i)  = Real_t(0.0);
        q(i)  = Real_t(0.0);
        ss(i) = Real_t(0.0);
    });

    // 初始体积 v = 1.0（重要：不能为 0）
    // Volume initial v = 1.0 (important : ne doit pas être zéro)
    Kokkos::parallel_for("InitElemVolume", numElem(), KOKKOS_LAMBDA(const Index_t i) {
        v(i) = Real_t(1.0);
    });

    // 节点速度初始化：xd, yd, zd = 0
    // Initialisation des vitesses nodales : xd, yd, zd = 0
    Kokkos::parallel_for("InitNodeVel", numNode(), KOKKOS_LAMBDA(const Index_t i) {
        xd(i) = Real_t(0.0);
        yd(i) = Real_t(0.0);
        zd(i) = Real_t(0.0);
    });

    // 节点加速度初始化：xdd, ydd, zdd = 0
    // Initialisation des accélérations nodales : xdd, ydd, zdd = 0
    Kokkos::parallel_for("InitNodeAcc", numNode(), KOKKOS_LAMBDA(const Index_t i) {
        xdd(i) = Real_t(0.0);
        ydd(i) = Real_t(0.0);
        zdd(i) = Real_t(0.0);
    });

    // 节点质量初始化：nodalMass = 0
    // Initialisation des masses nodales : nodalMass = 0
    Kokkos::parallel_for("InitNodeMass", numNode(), KOKKOS_LAMBDA(const Index_t i) {
        nodalMass(i) = Real_t(0.0);
    });

    // 构建网格：节点坐标、单元连接关系等
    // Construction du maillage : coordonnées nodales, connectivité des éléments, etc.
    BuildMesh(nx, edgeNodes, edgeElems);

    #if _OPENMP
    // 如果启用 OpenMP：创建线程支持结构（根据原始代码保持不动）
    // Si OpenMP est activé : création des structures de support de threads
    SetupThreadSupportStructures();
#else
    // 未使用 OpenMP：这些指针保持为 NULL
    // Sans OpenMP : ces pointeurs restent NULL
    m_nodeElemStart       = NULL;
    m_nodeElemCornerList  = NULL;
#endif

    // 创建区域索引集（regNumList）
    // Création des ensembles d'indices de régions
    CreateRegionIndexSets(nr, balance);

    // 设置对称平面（用于边界条件）
    // Configuration des plans de symétrie (conditions aux limites)
    SetupSymmetryPlanes(edgeNodes);

    // 设置单元连接关系（元素-节点）
    // Définition des connectivités élémentaires (élément → nœuds)
    SetupElementConnectivities(edgeElems);

    // 设置对称平面/自由面边界条件数组
    // Configuration des plans de symétrie / conditions de surface libre
    SetupBoundaryConditions(edgeElems);

        // 默认时间步配置（负数表示使用 Courant 条件）
    // Paramètres temporels par défaut (valeur négative → utilise la condition CFL)
    dtfixed() = Real_t(-1.0e-6);

    // 总模拟终止时间
    // Temps final de la simulation
    stoptime() = Real_t(1.0e-2);

    // 时间步调整上下限
    // Limites de variation du pas de temps
    deltatimemultlb() = Real_t(1.1);
    deltatimemultub() = Real_t(1.2);

    // Courant 与水动力约束初始为极大值
    // Contraintes CFL et hydrodynamiques initialisées à des valeurs énormes
    dtcourant() = Real_t(1.0e+20);
    dthydro()   = Real_t(1.0e+20);

    // 最大时间步
    // Pas de temps maximum
    dtmax() = Real_t(1.0e-2);

    // 起始时间、起始循环计数
    // Temps initial et compteur de cycle
    time()  = Real_t(0.0);
    cycle() = Int_t(0);

        // 根据当前网格与几何，计算每个单元的初始体积 volo 和单元质量 elemMass
    // Calcul des volumes initiaux volo et des masses élémentaires elemMass
    for (Index_t i = 0; i < numElem(); ++i) {
        Real_t x_local[8], y_local[8], z_local[8];
        Index_t* elemToNode = nodelist(i);

        // 收集单元节点坐标
        // Récupération des coordonnées nodales de l’élément
        for (Index_t lnode = 0; lnode < 8; ++lnode) {
            Index_t gnode = elemToNode[lnode];
            x_local[lnode] = x(gnode);
            y_local[lnode] = y(gnode);
            z_local[lnode] = z(gnode);
        }

        // 计算单元体积
        // Calcul du volume de l’élément
        Real_t volume = CalcElemVolume(x_local, y_local, z_local);
        volo(i) = volume;
        elemMass(i) = volume;

        // 平均分配节点质量（每单元 8 个节点）
        // Répartition uniforme de la masse aux 8 nœuds
        for (Index_t j = 0; j < 8; ++j) {
            Index_t idx = elemToNode[j];
            nodalMass(idx) += volume / Real_t(8.0);
        }
    }

        // 初始能量：按 Sedov 问题比例缩放
    // Dépôt d’énergie initial selon l’échelle du problème Sedov
    const Real_t ebase = Real_t(3.948746e+7);
    Real_t scale = (nx * m_tp) / Real_t(45.0);
    Real_t einit = ebase * scale * scale * scale;

    // 仅 rank(0,0,0) 在第 0 个单元放入能量冲击
    // Seul le rang (0,0,0) dépose l’énergie dans l’élément 0
    if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
        e(0) = einit;
    }

    // 初始时间步 = 0.5 * h / sqrt(2E)
    // Pas de temps initial basé sur CFL analytique : 0.5*h / sqrt(2E)
    deltatime() = (Real_t(.5) * cbrt(volo(0))) / sqrt(Real_t(2.0) * einit);
}
Domain::~Domain() {}

void Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  Index_t meshEdgeElems = m_tp * nx;

  // 初始化节点坐标
  // Initialisation des coordonnées nodales
  Index_t nidx = 0;
  Real_t tz = Real_t(1.125) * Real_t(m_planeLoc * nx) / Real_t(meshEdgeElems);

  for (Index_t plane = 0; plane < edgeNodes; ++plane) {
    Real_t ty = Real_t(1.125) * Real_t(m_rowLoc * nx) / Real_t(meshEdgeElems);
    for (Index_t row = 0; row < edgeNodes; ++row) {
      Real_t tx = Real_t(1.125) * Real_t(m_colLoc * nx) / Real_t(meshEdgeElems);
      for (Index_t col = 0; col < edgeNodes; ++col) {
        x(nidx) = tx;
        y(nidx) = ty;
        z(nidx) = tz;
        ++nidx;

        // 避免累积误差：直接计算下一列 tx
        // Pour éviter l’accumulation d’erreurs : tx recalculé directement
        tx = Real_t(1.125) * Real_t(m_colLoc * nx + col + 1) / Real_t(meshEdgeElems);
      }

      // 直接计算下一行 ty
      // Recalcul direct de ty
      ty = Real_t(1.125) * Real_t(m_rowLoc * nx + row + 1) / Real_t(meshEdgeElems);
    }

    // 直接计算下一层 tz
    // Recalcul direct de tz
    tz = Real_t(1.125) * Real_t(m_planeLoc * nx + plane + 1) / Real_t(meshEdgeElems);
  }

  // 构建六面体单元与节点的连接关系
  // Construction de la connectivité éléments–nœuds (hexaèdres)
  Index_t zidx = 0;
  nidx = 0;

  for (Index_t plane = 0; plane < edgeElems; ++plane) {
    for (Index_t row = 0; row < edgeElems; ++row) {
      for (Index_t col = 0; col < edgeElems; ++col) {

        Index_t* localNode = nodelist(zidx);

        localNode[0] = nidx;
        localNode[1] = nidx + 1;
        localNode[2] = nidx + edgeNodes + 1;
        localNode[3] = nidx + edgeNodes;
        localNode[4] = nidx + edgeNodes * edgeNodes;
        localNode[5] = nidx + edgeNodes * edgeNodes + 1;
        localNode[6] = nidx + edgeNodes * edgeNodes + edgeNodes + 1;
        localNode[7] = nidx + edgeNodes * edgeNodes + edgeNodes;

        ++zidx;
        ++nidx;
      }
      ++nidx;
    }
    nidx += edgeNodes;
  }
}

void Domain::SetupThreadSupportStructures()
{
#if _OPENMP
  Index_t numthreads = omp_get_max_threads();
#else
  Index_t numthreads = 1;
#endif

  if (numthreads > 1) {

    // 每个节点关联的单元角点数量
    // Nombre de coins d’éléments associés à chaque nœud
    Index_t* nodeElemCount = new Index_t[numNode()];
    for (Index_t i = 0; i < numNode(); ++i) nodeElemCount[i] = 0;

    // 统计每个节点出现的次数
    // Comptage des occurrences nodales
    for (Index_t i = 0; i < numElem(); ++i) {
      Index_t* nl = nodelist(i);
      for (Index_t j = 0; j < 8; ++j)
        ++nodeElemCount[nl[j]];
    }

    // 前缀和：确定每个节点的起始偏移
    // Préfixe : calcule l’offset de départ pour chaque nœud
    m_nodeElemStart = new Index_t[numNode() + 1];
    m_nodeElemStart[0] = 0;
    for (Index_t i = 1; i <= numNode(); ++i) {
      m_nodeElemStart[i] = m_nodeElemStart[i - 1] + nodeElemCount[i - 1];
    }

    // 存储每个节点关联的单元角点索引
    // Stocke les coins d’éléments associés aux nœuds
    m_nodeElemCornerList = new Index_t[m_nodeElemStart[numNode()]];

    // 重置统计数组
    // Réinitialisation du compteur
    for (Index_t i = 0; i < numNode(); ++i) nodeElemCount[i] = 0;

    // 填充 cornerList（每个节点对应的单元角点索引）
    // Remplissage de cornerList
    for (Index_t i = 0; i < numElem(); ++i) {
      Index_t* nl = nodelist(i);
      for (Index_t j = 0; j < 8; ++j) {
        Index_t m = nl[j];
        Index_t k = i * 8 + j;
        Index_t offset = m_nodeElemStart[m] + nodeElemCount[m];
        m_nodeElemCornerList[offset] = k;
        ++nodeElemCount[m];
      }
    }

    // 基本越界检查
    // Vérification de débordement
    Index_t clSize = m_nodeElemStart[numNode()];
    for (Index_t i = 0; i < clSize; ++i) {
      Index_t clv = m_nodeElemCornerList[i];
      if (clv < 0 || clv > numElem() * 8) {
        fprintf(stderr, "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
      }
    }

    delete[] nodeElemCount;
  }
  else {
    // 非线程环境下不使用这些结构
    // Non utilisé sans threading
    m_nodeElemStart = NULL;
    m_nodeElemCornerList = NULL;
  }
}

void Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // 分配足够大的缓冲区用于幽灵节点数据
  // Alloue un tampon suffisamment grand pour les données fantômes nodales
  Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ())) + 1;
  m_maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize * maxEdgeSize);
  m_maxEdgeSize  = CACHE_ALIGN_REAL(maxEdgeSize);

  // 默认情况下假设与六个相邻子域通信
  // Suppose une communication avec 6 voisins par défaut
  m_rowMin   = (m_rowLoc   == 0)        ? 0 : 1;
  m_rowMax   = (m_rowLoc   == m_tp - 1) ? 0 : 1;
  m_colMin   = (m_colLoc   == 0)        ? 0 : 1;
  m_colMax   = (m_colLoc   == m_tp - 1) ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)        ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp - 1) ? 0 : 1;

#if USE_MPI
  // 面通信大小
  // Taille de communication par faces
  Index_t comBufSize =
      (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
      m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM;

  // 棱通信大小
  // Taille de communication par arêtes
  comBufSize +=
      ((m_rowMin & m_colMin) +
       (m_rowMin & m_planeMin) +
       (m_colMin & m_planeMin) +
       (m_rowMax & m_colMax) +
       (m_rowMax & m_planeMax) +
       (m_colMax & m_planeMax) +
       (m_rowMax & m_colMin) +
       (m_rowMin & m_planeMax) +
       (m_colMin & m_planeMax) +
       (m_rowMin & m_colMax) +
       (m_rowMax & m_planeMin) +
       (m_colMax & m_planeMin)) *
      m_maxEdgeSize * MAX_FIELDS_PER_MPI_COMM;

  // 角通信大小（×16 以确保缓存行隔离）
  // Taille de communication par coins (×16 pour isoler les lignes de cache)
  comBufSize +=
      ((m_rowMin & m_colMin & m_planeMin) +
       (m_rowMin & m_colMin & m_planeMax) +
       (m_rowMin & m_colMax & m_planeMin) +
       (m_rowMin & m_colMax & m_planeMax) +
       (m_rowMax & m_colMin & m_planeMin) +
       (m_rowMax & m_colMin & m_planeMax) +
       (m_rowMax & m_colMax & m_planeMin) +
       (m_rowMax & m_colMax & m_planeMax)) *
      CACHE_COHERENCE_PAD_REAL;

  // 分配发送与接收缓冲区
  // Allocation des tampons d’envoi et de réception
  this->commDataSend = new Real_t[comBufSize];
  this->commDataRecv = new Real_t[comBufSize];

  // 防止浮点异常：全部置零
  // Pour éviter les exceptions flottantes : mise à zéro
  memset(this->commDataSend, 0, comBufSize * sizeof(Real_t));
  memset(this->commDataRecv, 0, comBufSize * sizeof(Real_t));
#endif

  // 边界节点集合（仅在对应物理边界处创建）
  // Ensembles de nœuds frontières (créés seulement sur les vraies frontières)
  if (m_colLoc == 0)
    m_symmX.resize(edgeNodes * edgeNodes);

  if (m_rowLoc == 0)
    m_symmY.resize(edgeNodes * edgeNodes);

  if (m_planeLoc == 0)
    m_symmZ.resize(edgeNodes * edgeNodes);
}

void Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI
  Index_t myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  // 用 MPI rank 初始化随机数种子
  // Initialise la graine aléatoire avec le rang MPI
  srand(myRank);
#else
  srand(0);
  Index_t myRank = 0;
#endif

  this->numReg() = nr;
  m_regElemSize = new Index_t[numReg()];
  m_regElemlist = new Index_t*[numReg()];

  Index_t nextIndex = 0;

  // 若只有一个区域，则所有单元都属于区域 1
  // Si une seule région, tous les éléments appartiennent à la région 1
  if (numReg() == 1) {
    while (nextIndex < numElem()) {
      this->regNumList(nextIndex) = 1;
      nextIndex++;
    }
    regElemSize(0) = 0;
  }

  // 多区域时，按权重随机分配区域
  // Pour plusieurs régions, répartition aléatoire pondérée
  else {
    Int_t regionNum;
    Int_t regionVar;
    Int_t lastReg = -1;
    Int_t binSize;
    Index_t elements;
    Index_t runto = 0;
    Int_t costDenominator = 0;

    Int_t* regBinEnd = new Int_t[numReg()];

    // 根据负载平衡参数 balance（-b 选项）建立各区域权重
    // Calcule les poids relatifs selon le paramètre balance (-b)
    for (Index_t i = 0; i < numReg(); ++i) {
      regElemSize(i) = 0;
      costDenominator += pow((i + 1), balance);
      regBinEnd[i] = costDenominator;
    }

    // 持续分配直到所有单元分配完
    // Continue jusqu’à ce que tous les éléments soient assignés
    while (nextIndex < numElem()) {

      regionVar = rand() % costDenominator;
      Index_t i = 0;
      while (regionVar >= regBinEnd[i])
        i++;

      // 根据 rank 做循环旋转，使不同 rank 具有不同主导区域
      // Rotation selon le rang MPI pour diversifier les régions dominantes
      regionNum = ((i + myRank) % numReg()) + 1;

      // 避免连续选择相同区域
      // Évite de sélectionner deux fois la même région
      while (regionNum == lastReg) {
        regionVar = rand() % costDenominator;
        i = 0;
        while (regionVar >= regBinEnd[i])
          i++;
        regionNum = ((i + myRank) % numReg()) + 1;
      }

      // 随机决定当前区域块的大小
      // Détermine aléatoirement la taille du bloc d’éléments
      binSize = rand() % 1000;
      if (binSize < 773) {
        elements = rand() % 15 + 1;
      }
      else if (binSize < 937) {
        elements = rand() % 16 + 16;
      }
      else if (binSize < 970) {
        elements = rand() % 32 + 32;
      }
      else if (binSize < 974) {
        elements = rand() % 64 + 64;
      }
      else if (binSize < 978) {
        elements = rand() % 128 + 128;
      }
      else if (binSize < 981) {
        elements = rand() % 256 + 256;
      }
      else {
        elements = rand() % 1537 + 512;
      }

      runto = elements + nextIndex;

      // 将当前区域的单元编号写入 regNumList
      // Assigne les éléments sélectionnés à la région choisie
      while (nextIndex < runto && nextIndex < numElem()) {
        this->regNumList(nextIndex) = regionNum;
        nextIndex++;
      }

      lastReg = regionNum;
    }
  }

  // 将 regNumList 转换为区域索引集（列表）
  // Convertit regNumList en listes d’index de régions

  // 统计每个区域大小
  // Compte la taille de chaque région
  for (Index_t i = 0; i < numElem(); ++i) {
    int r = this->regNumList(i) - 1;
    regElemSize(r)++;
  }

  // 为每个区域分配 index 列表
  // Alloue la liste d’index pour chaque région
  for (Index_t i = 0; i < numReg(); ++i) {
    m_regElemlist[i] = new Index_t[regElemSize(i)];
    regElemSize(i) = 0;
  }

  // 填充 index 列表
  // Remplit les listes d’index
  for (Index_t i = 0; i < numElem(); ++i) {
    Index_t r = regNumList(i) - 1;
    Index_t regndx = regElemSize(r)++;
    regElemlist(r, regndx) = i;
  }
}

void Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
  Index_t nidx = 0;

  // 遍历所有节点层
  // Parcourt toutes les couches de nœuds
  for (Index_t i = 0; i < edgeNodes; ++i) {

    Index_t planeInc = i * edgeNodes * edgeNodes;
    // 当前层的平面偏移
    // Décalage du plan pour cette couche

    Index_t rowInc = i * edgeNodes;
    // 当前层的行偏移
    // Décalage de ligne pour cette couche

    for (Index_t j = 0; j < edgeNodes; ++j) {

      if (m_planeLoc == 0) {
        m_symmZ[nidx] = rowInc + j;
        // Z=0 面的对称节点索引
        // Index du nœud symétrique sur le plan Z=0
      }

      if (m_rowLoc == 0) {
        m_symmY[nidx] = planeInc + j;
        // Y=0 面的对称节点索引
        // Index du nœud symétrique sur le plan Y=0
      }

      if (m_colLoc == 0) {
        m_symmX[nidx] = planeInc + j * edgeNodes;
        // X=0 面的对称节点索引
        // Index du nœud symétrique sur le plan X=0
      }

      ++nidx;
      // 增加全局节点编号
      // Incrémente l’index global du nœud
    }
  }
}

void Domain::SetupElementConnectivities(Int_t edgeElems)
{
  lxim(0) = 0;
  // 第一个单元在 X- 方向没有左邻居
  // Le premier élément n’a pas de voisin en direction X-

  for (Index_t i = 1; i < numElem(); ++i) {
    lxim(i) = i - 1;
    // 每个单元的 X- 邻居是前一个单元
    // Le voisin X- est l’élément précédent

    lxip(i - 1) = i;
    // 每个单元的 X+ 邻居是下一个单元
    // Le voisin X+ est l’élément suivant
  }

  lxip(numElem() - 1) = numElem() - 1;
  // 最后一个单元在 X+ 方向没有邻居
  // Le dernier élément n’a pas de voisin en direction X+


  // 设置 η（Y方向）邻居
  // Définit les voisins η (direction Y)
  for (Index_t i = 0; i < edgeElems; ++i) {
    letam(i) = i;
    // 最顶部一行没有 η− 邻居
    // La première ligne n’a pas de voisin en η−

    letap(numElem() - edgeElems + i) = numElem() - edgeElems + i;
    // 最底部一行没有 η+ 邻居
    // La dernière ligne n’a pas de voisin en η+
  }

  for (Index_t i = edgeElems; i < numElem(); ++i) {
    letam(i) = i - edgeElems;
    // η− 邻居为上方 edgeElems 之前
    // Le voisin η− est edgeElems éléments au-dessus

    letap(i - edgeElems) = i;
    // η+ 邻居为下方 edgeElems 之后
    // Le voisin η+ est edgeElems éléments en dessous
  }


  // 设置 ζ（Z方向）邻居
  // Définit les voisins ζ (direction Z)
  for (Index_t i = 0; i < edgeElems * edgeElems; ++i) {
    lzetam(i) = i;
    // 最顶部平面没有 ζ− 邻居
    // Le premier plan n’a pas de voisin en ζ−

    lzetap(numElem() - edgeElems * edgeElems + i) =
        numElem() - edgeElems * edgeElems + i;
    // 最底部平面没有 ζ+ 邻居
    // Le dernier plan n’a pas de voisin en ζ+
  }

  for (Index_t i = edgeElems * edgeElems; i < numElem(); ++i) {
    lzetam(i) = i - edgeElems * edgeElems;
    // ζ− 邻居为上一平面
    // Le voisin ζ− est un plan au-dessus

    lzetap(i - edgeElems * edgeElems) = i;
    // ζ+ 邻居为下一平面
    // Le voisin ζ+ est un plan en dessous
  }
}

void Domain::SetupBoundaryConditions(Int_t edgeElems)
{
  Index_t ghostIdx[6];  
  // ghostIdx：每个方向的 ghost 区在全局索引中的起始偏移
  // ghostIdx : décalage initial des zones fantômes dans chaque direction

  for (Index_t i = 0; i < numElem(); ++i) {
    elemBC(i) = Int_t(0);
    // 初始化所有单元的边界条件标志为 0
    // Initialise tous les indicateurs de condition aux limites à 0
  }

  for (Index_t i = 0; i < 6; ++i) {
    ghostIdx[i] = INT_MIN;
    // 默认 ghost 区索引设为 INT_MIN（无效）
    // Par défaut, l’index des zones fantômes est INT_MIN (invalide)
  }

  Int_t pidx = numElem();
  // pidx：ghost 区起始位置 = 本地单元数量之后
  // pidx : début des zones fantômes = après les éléments locaux

  if (m_planeMin != 0) {
    ghostIdx[0] = pidx;
    pidx += sizeX() * sizeY();
    // 如果存在下方平面邻域，则分配 ζ− ghost 平面
    // Si un voisin existe sur le plan inférieur, alloue le plan fantôme ζ−
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx;
    pidx += sizeX() * sizeY();
    // ζ+ ghost 平面
    // Plan fantôme ζ+
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx;
    pidx += sizeX() * sizeZ();
    // η− ghost 行
    // Ligne fantôme η−
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx;
    pidx += sizeX() * sizeZ();
    // η+ ghost 行
    // Ligne fantôme η+
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx;
    pidx += sizeY() * sizeZ();
    // ξ− ghost 列
    // Colonne fantôme ξ−
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx;
    // ξ+ ghost 列
    // Colonne fantôme ξ+
  }

  // 设置 6 个方向的对称面或通信面
  // Applique les plans de symétrie ou de communication
  for (Index_t i = 0; i < edgeElems; ++i) {

    Index_t planeInc = i * edgeElems * edgeElems;
    // 每一层平面偏移
    // Décalage du plan

    Index_t rowInc = i * edgeElems;
    // 每一层行偏移
    // Décalage de ligne

    for (Index_t j = 0; j < edgeElems; ++j) {

      // ζ− BC
      if (m_planeLoc == 0) {
        elemBC(rowInc + j) |= ZETA_M_SYMM;
        // 下平面在 Z=0，是对称面
        // Le plan inférieur Z=0 est un plan de symétrie
      }
      else {
        elemBC(rowInc + j) |= ZETA_M_COMM;
        lzetam(rowInc + j) = ghostIdx[0] + rowInc + j;
        // 否则为通信面，指向 ghost 区
        // Sinon, c’est une face de communication vers la zone fantôme
      }

      // ζ+ BC
      if (m_planeLoc == m_tp - 1) {
        elemBC(rowInc + j + numElem() - edgeElems * edgeElems) |= ZETA_P_FREE;
        // 上平面为自由边界
        // Le plan supérieur est une surface libre
      }
      else {
        elemBC(rowInc + j + numElem() - edgeElems * edgeElems) |= ZETA_P_COMM;
        lzetap(rowInc + j + numElem() - edgeElems * edgeElems) =
            ghostIdx[1] + rowInc + j;
        // 否则为通信面
        // Sinon, face de communication
      }

      // η− BC
      if (m_rowLoc == 0) {
        elemBC(planeInc + j) |= ETA_M_SYMM;
        // Y=0 面为对称面
        // Le plan Y=0 est un plan de symétrie
      }
      else {
        elemBC(planeInc + j) |= ETA_M_COMM;
        letam(planeInc + j) = ghostIdx[2] + rowInc + j;
        // 否则通信
        // Sinon, communication
      }

      // η+ BC
      if (m_rowLoc == m_tp - 1) {
        elemBC(planeInc + j + edgeElems * edgeElems - edgeElems) |= ETA_P_FREE;
        // 最后一行是自由边界
        // La dernière ligne est une surface libre
      }
      else {
        elemBC(planeInc + j + edgeElems * edgeElems - edgeElems) |= ETA_P_COMM;
        letap(planeInc + j + edgeElems * edgeElems - edgeElems) =
            ghostIdx[3] + rowInc + j;
      }

      // ξ− BC
      if (m_colLoc == 0) {
        elemBC(planeInc + j * edgeElems) |= XI_M_SYMM;
        // X=0 面对称
        // Le plan X=0 est symétrique
      }
      else {
        elemBC(planeInc + j * edgeElems) |= XI_M_COMM;
        lxim(planeInc + j * edgeElems) = ghostIdx[4] + rowInc + j;
      }

      // ξ+ BC
      if (m_colLoc == m_tp - 1) {
        elemBC(planeInc + j * edgeElems + edgeElems - 1) |= XI_P_FREE;
        // X 最大面是自由边界
        // Le plan X_max est une surface libre
      }
      else {
        elemBC(planeInc + j * edgeElems + edgeElems - 1) |= XI_P_COMM;
        lxip(planeInc + j * edgeElems + edgeElems - 1) =
            ghostIdx[5] + rowInc + j;
      }
    }
  }
}

void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
  Int_t testProcs;
  Int_t dx, dy, dz;
  Int_t myDom;

  // 当前假设处理器以立方体方式划分
  // Supposition actuelle : découpage des processeurs en forme cubique
  testProcs = Int_t(cbrt(Real_t(numRanks)) + 0.5);

  if (testProcs * testProcs * testProcs != numRanks) {
    printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n");
    // 处理器数量必须是整数的立方数
    // Le nombre de processeurs doit être un cube parfait

#if USE_MPI
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    exit(-1);
#endif
  }

  if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
    printf("MPI operations only support float and double right now...\n");
    // MPI 目前仅支持 float 与 double
    // MPI ne supporte actuellement que float et double

#if USE_MPI
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    exit(-1);
#endif
  }

  if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
    printf("corner element comm buffers too small.  Fix code.\n");
    // 角点通信缓冲区过小，需要修复代码
    // Les buffers de communication des coins sont trop petits : corriger le code

#if USE_MPI
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    exit(-1);
#endif
  }

  dx = testProcs;
  // x 方向处理器数
  // Nombre de processeurs en direction x

  dy = testProcs;
  // y 方向处理器数
  // Nombre de processeurs en direction y

  dz = testProcs;
  // z 方向处理器数
  // Nombre de processeurs en direction z

  if (dx * dy * dz != numRanks) {
    printf("error -- must have as many domains as procs\n");
    // 错误：域数量必须等于处理器数量
    // Erreur : le nombre de domaines doit égaler le nombre de processeurs

#if USE_MPI
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    exit(-1);
#endif
  }

  Int_t remainder = dx * dy * dz % numRanks;
  // 多域分配时的余数
  // Reste lors de la distribution des domaines

  if (myRank < remainder) {
    myDom = myRank * (1 + (dx * dy * dz / numRanks));
    // 前 remainder 个 rank 分配多一个域
    // Les premiers rangs reçoivent un domaine supplémentaire
  }
  else {
    myDom = remainder * (1 + (dx * dy * dz / numRanks))
          + (myRank - remainder) * (dx * dy * dz / numRanks);
    // 其余 rank 分配标准数量的域
    // Les autres rangs reçoivent leur part standard
  }

  *col = myDom % dx;
  // 计算该子域的 x 方向索引
  // Calcule l’indice x du sous-domaine

  *row = (myDom / dx) % dy;
  // 计算该子域的 y 方向索引
  // Calcule l’indice y du sous-domaine

  *plane = myDom / (dx * dy);
  // 计算该子域的 z 方向索引
  // Calcule l’indice z du sous-domaine

  *side = testProcs;
  // side：立方划分的边长
  // side : longueur du côté du découpage cubique

  return;
}
