#if !defined(USE_MPI)
#error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

// --- Gestion OpenMP optionnelle
// --- 可选 OpenMP 支持（最终由 Kokkos 管理）
#if USE_MPI
#include <mpi.h>

/*
   Définir un des symboles suivants :
   请选择以下符号之一定义同步策略：

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

// --- Inclusions Kokkos requises (Kokkos 5.0)
// --- 包含 Kokkos 5.0 所需头文件
#include <Kokkos_Core.hpp>

#include <cmath>
#include <vector>

// --- Définition simple du macro MAX
// --- MAX 宏定义
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// --- Spécifications des précisions
// --- 精度类型定义
typedef float       real4;
typedef double      real8;
typedef long double real10;

typedef int    Index_t;  // index
typedef real8  Real_t;   // type réel
typedef int    Int_t;    // type entier

enum { VolumeError = -1, QStopError = -2 };

// --- Surcharge des fonctions mathématiques
// --- 数学函数重载
inline real4  SQRT(real4 arg)  { return sqrtf(arg); }
inline real8  SQRT(real8 arg)  { return sqrt(arg); }
inline real10 SQRT(real10 arg) { return sqrtl(arg); }

inline real4  CBRT(real4 arg)  { return cbrtf(arg); }
inline real8  CBRT(real8 arg)  { return cbrt(arg); }
inline real10 CBRT(real10 arg) { return cbrtl(arg); }

inline real4  FABS(real4 arg)  { return fabsf(arg); }
inline real8  FABS(real8 arg)  { return fabs(arg); }
inline real10 FABS(real10 arg) { return fabsl(arg); }

// --- Bits pour les conditions de frontière
// --- 边界条件位标志
#define XI_M       0x00007
#define XI_M_SYMM  0x00001
#define XI_M_FREE  0x00002
#define XI_M_COMM  0x00004

#define XI_P       0x00038
#define XI_P_SYMM  0x00008
#define XI_P_FREE  0x00010
#define XI_P_COMM  0x00020

#define ETA_M      0x001c0
#define ETA_M_SYMM 0x00040
#define ETA_M_FREE 0x00080
#define ETA_M_COMM 0x00100

#define ETA_P      0x00e00
#define ETA_P_SYMM 0x00200
#define ETA_P_FREE 0x00400
#define ETA_P_COMM 0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// --- Tags MPI
// --- MPI 标签
#define MSG_COMM_SBN       1024
#define MSG_SYNC_POS_VEL   2048
#define MSG_MONOQ          3072

#define MAX_FIELDS_PER_MPI_COMM 6

// --- Alignement pour cohérence cache
// --- 缓存一致性填充
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))
#define CACHE_ALIGN_REAL(n) (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL - 1))

// --- Début de la structure principale Domain
// --- 主数据结构 Domain 类开始
class Domain {

public:

  // --- Constructeur
  // --- 构造函数（后续你会发送实现部分，我再同步迁移）
  Domain(Int_t numRanks,
         Index_t colLoc, Index_t rowLoc, Index_t planeLoc,
         Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

  // --- Destructeur
  // --- 析构函数
  ~Domain() = default;

  // ALLOCATION DES DONNÉES (version Kokkos::View)
  // 数据分配（使用 Kokkos::View）

  // --- Allocation des données nodales
  // --- 节点相关数据分配
  void AllocateNodePersistent(Int_t numNode)
  {
    // Coordonnées
    m_x  = Kokkos::View<Real_t*>("m_x",  numNode);
    m_y  = Kokkos::View<Real_t*>("m_y",  numNode);
    m_z  = Kokkos::View<Real_t*>("m_z",  numNode);

    // Vitesse
    m_xd = Kokkos::View<Real_t*>("m_xd", numNode);
    m_yd = Kokkos::View<Real_t*>("m_yd", numNode);
    m_zd = Kokkos::View<Real_t*>("m_zd", numNode);

    // Accélération
    m_xdd = Kokkos::View<Real_t*>("m_xdd", numNode);
    m_ydd = Kokkos::View<Real_t*>("m_ydd", numNode);
    m_zdd = Kokkos::View<Real_t*>("m_zdd", numNode);

    // Forces nodales
    m_fx = Kokkos::View<Real_t*>("m_fx", numNode);
    m_fy = Kokkos::View<Real_t*>("m_fy", numNode);
    m_fz = Kokkos::View<Real_t*>("m_fz", numNode);

    // Masse nodale
    m_nodalMass = Kokkos::View<Real_t*>("m_nodalMass", numNode);
  }

  // --- Allocation des données élément-centrées
  // --- 单元相关数据分配
  void AllocateElemPersistent(Int_t numElem)
  {
    // Connectivité (8 nœuds par élément)
    m_nodelist = Kokkos::View<Index_t*>("m_nodelist", 8 * numElem);

    // Connectivités entre faces
    m_lxim   = Kokkos::View<Index_t*>("m_lxim",   numElem);
    m_lxip   = Kokkos::View<Index_t*>("m_lxip",   numElem);
    m_letam  = Kokkos::View<Index_t*>("m_letam",  numElem);
    m_letap  = Kokkos::View<Index_t*>("m_letap",  numElem);
    m_lzetam = Kokkos::View<Index_t*>("m_lzetam", numElem);
    m_lzetap = Kokkos::View<Index_t*>("m_lzetap", numElem);

    m_elemBC = Kokkos::View<Int_t*>("m_elemBC", numElem);

    // Énergie / Pression
    m_e = Kokkos::View<Real_t*>("m_e", numElem);
    m_p = Kokkos::View<Real_t*>("m_p", numElem);

    // Monotonic Q
    m_q  = Kokkos::View<Real_t*>("m_q",  numElem);
    m_ql = Kokkos::View<Real_t*>("m_ql", numElem);
    m_qq = Kokkos::View<Real_t*>("m_qq", numElem);

    // Volume
    m_v    = Kokkos::View<Real_t*>("m_v",    numElem);
    m_volo = Kokkos::View<Real_t*>("m_volo", numElem);
    m_delv = Kokkos::View<Real_t*>("m_delv", numElem);
    m_vdov = Kokkos::View<Real_t*>("m_vdov", numElem);

    // Aire réelle
    m_arealg = Kokkos::View<Real_t*>("m_arealg", numElem);

    // Sound speed
    m_ss = Kokkos::View<Real_t*>("m_ss", numElem);

    // Masse élément
    m_elemMass = Kokkos::View<Real_t*>("m_elemMass", numElem);

    // Nouveau volume
    m_vnew = Kokkos::View<Real_t*>("m_vnew", numElem);
  }

  // --- Allocation des gradients
  // --- 梯度数据分配
  void AllocateGradients(Int_t numElem, Int_t allElem)
  {
    m_delx_xi   = Kokkos::View<Real_t*>("m_delx_xi",   numElem);
    m_delx_eta  = Kokkos::View<Real_t*>("m_delx_eta",  numElem);
    m_delx_zeta = Kokkos::View<Real_t*>("m_delx_zeta", numElem);

    m_delv_xi   = Kokkos::View<Real_t*>("m_delv_xi",   allElem);
    m_delv_eta  = Kokkos::View<Real_t*>("m_delv_eta",  allElem);
    m_delv_zeta = Kokkos::View<Real_t*>("m_delv_zeta", allElem);
  }

  // --- Libération (View 没有 clear，需要重置为空 View)
  void DeallocateGradients()
  {
    m_delx_xi   = Kokkos::View<Real_t*>();
    m_delx_eta  = Kokkos::View<Real_t*>();
    m_delx_zeta = Kokkos::View<Real_t*>();

    m_delv_xi   = Kokkos::View<Real_t*>();
    m_delv_eta  = Kokkos::View<Real_t*>();
    m_delv_zeta = Kokkos::View<Real_t*>();
  }

  // --- Strains
  void AllocateStrains(Int_t numElem)
  {
    m_dxx = Kokkos::View<Real_t*>("m_dxx", numElem);
    m_dyy = Kokkos::View<Real_t*>("m_dyy", numElem);
    m_dzz = Kokkos::View<Real_t*>("m_dzz", numElem);
  }

  void DeallocateStrains()
  {
    m_dxx = Kokkos::View<Real_t*>();
    m_dyy = Kokkos::View<Real_t*>();
    m_dzz = Kokkos::View<Real_t*>();
  }

// --- ACCESSORS : version Kokkos 5.0
// --- 访问器：Kokkos 5.0 风格

// --- Informations de région
// --- 区域信息
KOKKOS_INLINE_FUNCTION
Index_t regElemSize(Index_t idx) const { return m_regElemSize(idx); }

KOKKOS_INLINE_FUNCTION
Index_t regNumList(Index_t idx) const { return m_regNumList(idx); }

// --- 这些原始指针访问方式已不再适用，在后续重写 regElemlist 结构
// --- 下方 regElemlist 的存储方式需要重构（不同文件中关联）
KOKKOS_INLINE_FUNCTION
Index_t regElemlist(Int_t r, Index_t idx) const {
    return m_regElemlist(r, idx);
}

KOKKOS_INLINE_FUNCTION
Index_t nodelist(Index_t elem, Int_t local) const {
    return m_nodelist(elem * 8 + local);
}

// --- Connectivités entre faces
// --- 面的连接关系
KOKKOS_INLINE_FUNCTION Index_t lxim(Index_t idx) const   { return m_lxim(idx); }
KOKKOS_INLINE_FUNCTION Index_t lxip(Index_t idx) const   { return m_lxip(idx); }
KOKKOS_INLINE_FUNCTION Index_t letam(Index_t idx) const  { return m_letam(idx); }
KOKKOS_INLINE_FUNCTION Index_t letap(Index_t idx) const  { return m_letap(idx); }
KOKKOS_INLINE_FUNCTION Index_t lzetam(Index_t idx) const { return m_lzetam(idx); }
KOKKOS_INLINE_FUNCTION Index_t lzetap(Index_t idx) const { return m_lzetap(idx); }

// --- Conditions de frontière des éléments
// --- 单元边界条件标志
KOKKOS_INLINE_FUNCTION Int_t elemBC(Index_t idx) const { return m_elemBC(idx); }

// --- Déformations principales
// --- 主应变（临时量）
KOKKOS_INLINE_FUNCTION Real_t dxx(Index_t idx) const { return m_dxx(idx); }
KOKKOS_INLINE_FUNCTION Real_t dyy(Index_t idx) const { return m_dyy(idx); }
KOKKOS_INLINE_FUNCTION Real_t dzz(Index_t idx) const { return m_dzz(idx); }

// --- Nouveau volume relatif
// --- 新相对体积
KOKKOS_INLINE_FUNCTION Real_t vnew(Index_t idx) const { return m_vnew(idx); }

// --- Gradient de vitesse
// --- 速度梯度
KOKKOS_INLINE_FUNCTION Real_t delv_xi(Index_t idx) const   { return m_delv_xi(idx); }
KOKKOS_INLINE_FUNCTION Real_t delv_eta(Index_t idx) const  { return m_delv_eta(idx); }
KOKKOS_INLINE_FUNCTION Real_t delv_zeta(Index_t idx) const { return m_delv_zeta(idx); }

// --- Gradient de position
// --- 坐标梯度
KOKKOS_INLINE_FUNCTION Real_t delx_xi(Index_t idx) const   { return m_delx_xi(idx); }
KOKKOS_INLINE_FUNCTION Real_t delx_eta(Index_t idx) const  { return m_delx_eta(idx); }
KOKKOS_INLINE_FUNCTION Real_t delx_zeta(Index_t idx) const { return m_delx_zeta(idx); }

// --- Énergie
KOKKOS_INLINE_FUNCTION Real_t e(Index_t idx) const { return m_e(idx); }

// --- Pression
KOKKOS_INLINE_FUNCTION Real_t p(Index_t idx) const { return m_p(idx); }

// --- Viscosité artificielle
KOKKOS_INLINE_FUNCTION Real_t q(Index_t idx) const { return m_q(idx); }

// --- Termes linéaire et quadratique de q
KOKKOS_INLINE_FUNCTION Real_t ql(Index_t idx) const { return m_ql(idx); }
KOKKOS_INLINE_FUNCTION Real_t qq(Index_t idx) const { return m_qq(idx); }

// --- Volume relatif
KOKKOS_INLINE_FUNCTION Real_t v(Index_t idx) const { return m_v(idx); }

// --- Variation du volume
KOKKOS_INLINE_FUNCTION Real_t delv(Index_t idx) const { return m_delv(idx); }

// --- Volume de référence
KOKKOS_INLINE_FUNCTION Real_t volo(Index_t idx) const { return m_volo(idx); }

// --- dérivée de volume
KOKKOS_INLINE_FUNCTION Real_t vdov(Index_t idx) const { return m_vdov(idx); }

// --- Longueur caractéristique
KOKKOS_INLINE_FUNCTION Real_t arealg(Index_t idx) const { return m_arealg(idx); }

// --- Vitesse du son
KOKKOS_INLINE_FUNCTION Real_t ss(Index_t idx) const { return m_ss(idx); }

// --- Masse élément
KOKKOS_INLINE_FUNCTION Real_t elemMass(Index_t idx) const {
    return m_elemMass(idx);
}

// --- Nombre d'éléments adjacents à un nœud
KOKKOS_INLINE_FUNCTION
Index_t nodeElemCount(Index_t idx) const {
    return m_nodeElemStart(idx + 1) - m_nodeElemStart(idx);
}

KOKKOS_INLINE_FUNCTION
Index_t nodeElemCornerList(Index_t idx, Index_t off) const {
    return m_nodeElemCornerList(m_nodeElemStart(idx) + off);
}

private:

  // --- Méthodes internes (inchangées)
  // --- 内部方法声明（保持不变）
  void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(Int_t nreg, Int_t balance);
  void SetupCommBuffers(Int_t edgeNodes);
  void SetupSymmetryPlanes(Int_t edgeNodes);
  void SetupElementConnectivities(Int_t edgeElems);
  void SetupBoundaryConditions(Int_t edgeElems);

  // --- IMPLEMENTATION : stockage des données par Kokkos::View
  // --- 实现部分：全部数据改为 Kokkos::View

  //  Node-centered data
  //  节点相关变量

  // --- Coordonnées nodales
  // --- 节点坐标
  Kokkos::View<Real_t*> m_x;
  Kokkos::View<Real_t*> m_y;
  Kokkos::View<Real_t*> m_z;

  // --- Vitesse nodale
  // --- 节点速度
  Kokkos::View<Real_t*> m_xd;
  Kokkos::View<Real_t*> m_yd;
  Kokkos::View<Real_t*> m_zd;

  // --- Accélération nodale
  // --- 节点加速度
  Kokkos::View<Real_t*> m_xdd;
  Kokkos::View<Real_t*> m_ydd;
  Kokkos::View<Real_t*> m_zdd;

  // --- Forces nodales
  // --- 节点力
  Kokkos::View<Real_t*> m_fx;
  Kokkos::View<Real_t*> m_fy;
  Kokkos::View<Real_t*> m_fz;

  // --- Masse nodale
  // --- 节点质量
  Kokkos::View<Real_t*> m_nodalMass;

  // --- Plan de symétrie
  // --- 对称面节点
  Kokkos::View<Index_t*> m_symmX;
  Kokkos::View<Index_t*> m_symmY;
  Kokkos::View<Index_t*> m_symmZ;

  //  Element-centered data
  //  单元相关变量

  // --- Région (information par région)
  // --- 区域信息
  Int_t m_numReg;     // nombre de régions / 区域数量
  Int_t m_cost;       // coût d'imbalance / 不平衡代价

  // --- 尺寸与区域编号列表（二维 view）
  // --- 重要：这里按照方案 A 使用二维 View
  Kokkos::View<Index_t*>  m_regElemSize;   // taille des régions / 每个区域包含的单元数
  Kokkos::View<Index_t*>  m_regNumList;    // numéro de région pour chaque élément / 每个单元所属区域编号
  Kokkos::View<Index_t**> m_regElemlist;   // liste des éléments par région / 每个区域的单元列表 (二维)

  // --- Connectivité élément→nœud（8 节点单元）
  Kokkos::View<Index_t*> m_nodelist;

  // --- Connectivité entre faces
  // --- 单元邻居（六个方向）
  Kokkos::View<Index_t*> m_lxim;
  Kokkos::View<Index_t*> m_lxip;
  Kokkos::View<Index_t*> m_letam;
  Kokkos::View<Index_t*> m_letap;
  Kokkos::View<Index_t*> m_lzetam;
  Kokkos::View<Index_t*> m_lzetap;

  // --- Conditions de frontière d’élément
  // --- 单元边界条件标志
  Kokkos::View<Int_t*> m_elemBC;

  // --- Déformations principales (temporaire)
  // --- 主应变（临时）
  Kokkos::View<Real_t*> m_dxx;
  Kokkos::View<Real_t*> m_dyy;
  Kokkos::View<Real_t*> m_dzz;

  // --- Gradient de vitesse
  // --- 速度梯度
  Kokkos::View<Real_t*> m_delv_xi;
  Kokkos::View<Real_t*> m_delv_eta;
  Kokkos::View<Real_t*> m_delv_zeta;

  // --- Gradient de position
  // --- 坐标梯度
  Kokkos::View<Real_t*> m_delx_xi;
  Kokkos::View<Real_t*> m_delx_eta;
  Kokkos::View<Real_t*> m_delx_zeta;

  // --- Énergie interne
  Kokkos::View<Real_t*> m_e;

  // --- Pression
  Kokkos::View<Real_t*> m_p;

  // --- q-viscosité artificielle
  Kokkos::View<Real_t*> m_q;
  Kokkos::View<Real_t*> m_ql;
  Kokkos::View<Real_t*> m_qq;

  // --- Volumes
  Kokkos::View<Real_t*> m_v;     // volume relatif
  Kokkos::View<Real_t*> m_volo;  // volume de référence
  Kokkos::View<Real_t*> m_vnew;  // nouveau volume
  Kokkos::View<Real_t*> m_delv;  // variation du volume
  Kokkos::View<Real_t*> m_vdov;  // dérivée de volume

  // --- Longueur caractéristique d’élément
  Kokkos::View<Real_t*> m_arealg;

  // --- Vitesse du son
  Kokkos::View<Real_t*> m_ss;

  // --- Masse de l’élément
  Kokkos::View<Real_t*> m_elemMass;

  //   Node → element adjacency
  //   节点对应的单元角点列表

  Kokkos::View<Index_t*> m_nodeElemStart;
  Kokkos::View<Index_t*> m_nodeElemCornerList;

  //  Constantes globales（全部保留原逻辑）
  //  全局常数（与你原版本完全一致）

  const Real_t m_e_cut;
  const Real_t m_p_cut;
  const Real_t m_q_cut;
  const Real_t m_v_cut;
  const Real_t m_u_cut;

  const Real_t m_hgcoef;
  const Real_t m_ss4o3;
  const Real_t m_qstop;
  const Real_t m_monoq_max_slope;
  const Real_t m_monoq_limiter_mult;
  const Real_t m_qlc_monoq;
  const Real_t m_qqc_monoq;
  const Real_t m_qqc;
  const Real_t m_eosvmax;
  const Real_t m_eosvmin;
  const Real_t m_pmin;
  const Real_t m_emin;
  const Real_t m_dvovmax;
  const Real_t m_refdens;

  // --- Variables du contrôle du pas de temps
  // --- 时间步控制相关变量
  Real_t m_dtcourant;   // limite de Courant / Courant 约束
  Real_t m_dthydro;     // limite de changement de volume / 体积变化约束
  Int_t  m_cycle;       // itération / 迭代编号
  Real_t m_dtfixed;     // pas fixe / 固定时间步（若启用）
  Real_t m_time;        // temps actuel / 当前时间
  Real_t m_deltatime;   // pas variable / 可变时间步
  Real_t m_deltatimemultlb; // multiplicateur inférieur / 下限因子
  Real_t m_deltatimemultub; // multiplicateur supérieur / 上限因子
  Real_t m_dtmax;        // pas maximal / 最大允许时间步
  Real_t m_stoptime;     // fin de la simulation / 模拟终止时间

  // --- Topologie du domaine parallèle
  // --- 并行域分解拓扑
  Int_t   m_numRanks;
  Index_t m_colLoc;
  Index_t m_rowLoc;
  Index_t m_planeLoc;
  Index_t m_tp;

  // --- Dimensions globales
  // --- 全局尺寸
  Index_t m_sizeX;
  Index_t m_sizeY;
  Index_t m_sizeZ;
  Index_t m_numElem;
  Index_t m_numNode;

  Index_t m_maxPlaneSize;
  Index_t m_maxEdgeSize;

  // --- Tables nodeElemStart et nodeElemCornerList (ancienne version = pointeurs)
  // --- 全部迁移为 Kokkos::View（在上一部分已迁移）
  //
  //    原始代码这里重复出现了指针版本
  //    为避免冲突，将其完全替换为 Kokkos::View（变量名保持不变）
  //
  //    原始：
  //        Index_t *m_nodeElemStart;
  //        Index_t *m_nodeElemCornerList;
  //
  //    迁移版本（上段已声明，这里不再重复声明）：
  //        Kokkos::View<Index_t*> m_nodeElemStart;
  //        Kokkos::View<Index_t*> m_nodeElemCornerList;
  //
  // --- 因此这里删除原始指针定义（避免重复定义错误）

  // --- Limites spatiales du sous-domaine
  // --- 子域空间范围
  Index_t m_rowMin,  m_rowMax;
  Index_t m_colMin,  m_colMax;
  Index_t m_planeMin, m_planeMax;

};  // --- Fin de la classe Domain / Domain 类结束


// --- Fonction pointeur Domain_member
// --- 类成员函数指针类型（用于通信模块）

typedef Real_t (Domain::*Domain_member)(Index_t) const;


// --- Options de la ligne de commande
// --- 命令行参数
struct cmdLineOpts {
  Int_t its;      // -i nombre d'itérations / 迭代次数
  Int_t nx;       // -s taille / 尺寸
  Int_t numReg;   // -r nombre de régions / 区域数量
  Int_t numFiles; // -f nombre de fichiers / 输出文件数
  Int_t showProg; // -p afficher progrès / 显示进度
  Int_t quiet;    // -q silencieux / 安静模式
  Int_t viz;      // -v visualisation / 可视化
  Int_t cost;     // -c coût d'imbalance / 负载成本
  Int_t balance;  // -b équilibrage / 负载平衡
};


// --- Déclarations de fonctions globales
// --- 全局函数声明

// lulesh-par
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char* argv[],
                             Int_t myRank, struct cmdLineOpts* opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx, Int_t numRanks);

// lulesh-viz
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain& domain, Int_t msgType,
              Index_t xferFields, Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);

void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields,
              Domain_member* fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);

void CommSBN(Domain& domain,
             Int_t xferFields,
             Domain_member* fieldData);

void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// lulesh-init
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t* col, Int_t* row, Int_t* plane, Int_t* side);


// --- MinFinder : Functor pour réduction parallèle (min)
// --- MinFinder：用于 parallel_reduce 的最小值 functor

struct MinFinder {
  Real_t val;
  int    i;

  KOKKOS_INLINE_FUNCTION
  MinFinder()
    : val(1.0e20), i(-1) // valeur très grande par défaut / 默认超大值
  {}

  KOKKOS_INLINE_FUNCTION
  MinFinder(const double& v, const int& idx)
    : val(v), i(idx)
  {}

  KOKKOS_INLINE_FUNCTION
  MinFinder(const MinFinder& src)
    : val(src.val), i(src.i)
  {}

  // --- Réduction + opérateur
  // --- 并行归约 += 运算符
  KOKKOS_INLINE_FUNCTION
  void operator+=(MinFinder& src) {
    if (src.val < val) {
      val = src.val;
      i = src.i;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile MinFinder& src) volatile {
    if (src.val < val) {
      val = src.val;
      i = src.i;
    }
  }
};
