#ifndef LULESH_KOKKOS_H
#define LULESH_KOKKOS_H

/* 
 * Inclusion de Kokkos 5.0 / 引入 Kokkos 5.0
 */
#include <Kokkos_Core.hpp>

/*
 * Définition : cette version force USE_MPI = 0 / 本版本强制关闭 MPI（S1 要求）
 */
#if !defined(USE_MPI)
#define USE_MPI 0
#endif

#if USE_MPI
#include <mpi.h>
#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

/* 
 * MAX macro — retourne la valeur max / 返回最大值 
 */
#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

/* 
 * Types numériques / 数值类型定义 
 */
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;

typedef int32_t Int4_t ;
typedef int64_t Int8_t ;
typedef Int4_t  Index_t ; /* index de boucle / 下标与循环索引 */
typedef real8   Real_t ;  /* type flottant principal / 主浮点类型 */
typedef Int4_t  Int_t ;   /* type entier / 整型 */

enum { VolumeError = -1, QStopError = -2 } ;

/* 
 * Fonctions mathématiques inline / 内联数学函数 
 */
inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline real4  FABS(real4  arg) { return fabsf(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real10 FABS(real10 arg) { return fabsl(arg) ; }

/* 
 * Codes de conditions limites / 边界条件编码（位掩码）
 * 6 faces * 2 conditions = 12 bits 
 */
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

/* 
 * Hypothèse de cohérence cache / 缓存一致性假设 
 */
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))
#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

/*
 * Anciennes fonctions d'allocation — inutiles avec Kokkos::View 
 * 旧的分配函数（Kokkos::View 版本不再使用，但为了兼容仍保留）
 */
template <typename T>
T *Allocate(size_t size)
{
   return static_cast<T *>(malloc(sizeof(T)*size)) ;
}

template <typename T>
void Release(T **ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

/*
 * Déclaration de la structure principale `Domain`
 * L'implémentation abstraite des données LULESH est contenue ici.
 * 主数据结构 Domain，封装 LULESH 中所有网格与物理场信息。
 */
class Domain {

public:

   /* 
    * Constructeur et destructeur / 构造与析构 
    */
   Domain(Int_t numRanks, Index_t colLoc,
          Index_t rowLoc, Index_t planeLoc,
          Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

   ~Domain();
   //
   // ALLOCATION — version Kokkos::View
   // 分配函数 — 使用 Kokkos::View 替代 std::vector / malloc
   //

   /* 
    * Allocation des champs nodaux persistants / 分配持久化节点字段 
    * Remplace std::vector::resize par Kokkos::View / 用 Kokkos::View 替代 vector
    */
   void SetupInitialVolumesAndMasses();
  
// ---- View accessors (write in Kokkos kernels)
KOKKOS_INLINE_FUNCTION
auto vdov_view() const { return m_vdov; }

auto v_view() const { return m_v; }
// Volume de référence
KOKKOS_INLINE_FUNCTION
auto volo_view() const { return m_volo; }

// Connectivité élément → nœuds
KOKKOS_INLINE_FUNCTION
auto nodelist_view() const { return m_nodelist; }

// Longueurs caractéristiques directionnelles
KOKKOS_INLINE_FUNCTION
auto delx_zeta_view() const { return m_delx_zeta; }

KOKKOS_INLINE_FUNCTION
auto delx_xi_view() const { return m_delx_xi; }

KOKKOS_INLINE_FUNCTION
auto delx_eta_view() const { return m_delx_eta; }


// Gradients de vitesse directionnels
KOKKOS_INLINE_FUNCTION
auto delv_zeta_view() const { return m_delv_zeta; }

KOKKOS_INLINE_FUNCTION
auto delv_xi_view() const { return m_delv_xi; }

KOKKOS_INLINE_FUNCTION
auto delv_eta_view() const { return m_delv_eta; }

// -------- Region / topology --------
Kokkos::View<Index_t**> regElemlist_view() const { return m_regElemlist; }
Kokkos::View<Int_t*>    elemBC_view()      const { return m_elemBC; }

// -------- Neighbors --------
Kokkos::View<Index_t*> lxim_view()   const { return m_lxim; }
Kokkos::View<Index_t*> lxip_view()   const { return m_lxip; }
Kokkos::View<Index_t*> letam_view()  const { return m_letam; }
Kokkos::View<Index_t*> letap_view()  const { return m_letap; }
Kokkos::View<Index_t*> lzetam_view() const { return m_lzetam; }
Kokkos::View<Index_t*> lzetap_view() const { return m_lzetap; }


Kokkos::View<Real_t*> ql_view() const { return m_ql; }
Kokkos::View<Real_t*> qq_view() const { return m_qq; }


Kokkos::View<Real_t*> e_view()    const { return m_e; }
Kokkos::View<Real_t*> delv_view() const { return m_delv; }
Kokkos::View<Real_t*> p_view()    const { return m_p; }
Kokkos::View<Real_t*> q_view()    const { return m_q; }

Kokkos::View<Real_t*> vnew_view()   const { return m_vnew; }
Kokkos::View<Real_t*> arealg_view() const { return m_arealg; }
Kokkos::View<Real_t*> dxx_view()    const { return m_dxx; }
Kokkos::View<Real_t*> dyy_view()    const { return m_dyy; }
Kokkos::View<Real_t*> dzz_view()    const { return m_dzz; }


 
   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Index_t*> symmX_view()const { return m_symmX; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Index_t*> symmY_view()const { return m_symmY; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Index_t*> symmZ_view()const { return m_symmZ; }

// --- Views getters for Kokkos kernels / 并行核使用的 View 接口
KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> xd_view()  const { return m_xd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> yd_view()  const { return m_yd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> zd_view()  const { return m_zd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> xdd_view() const { return m_xdd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> ydd_view() const { return m_ydd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> zdd_view() const { return m_zdd; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> x_view() const { return m_x; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> y_view() const { return m_y; }

KOKKOS_INLINE_FUNCTION
Kokkos::View<Real_t*> z_view() const { return m_z; }



   void AllocateNodePersistent(Int_t numNode)
   {
      // Coordonnées nodales / 节点坐标
      m_x = Kokkos::View<Real_t*>("x", numNode);
      m_y = Kokkos::View<Real_t*>("y", numNode);
      m_z = Kokkos::View<Real_t*>("z", numNode);

      // Vitesses nodales / 节点速度
      m_xd = Kokkos::View<Real_t*>("xd", numNode);
      m_yd = Kokkos::View<Real_t*>("yd", numNode);
      m_zd = Kokkos::View<Real_t*>("zd", numNode);

      // Accélérations nodales / 节点加速度
      m_xdd = Kokkos::View<Real_t*>("xdd", numNode);
      m_ydd = Kokkos::View<Real_t*>("ydd", numNode);
      m_zdd = Kokkos::View<Real_t*>("zdd", numNode);

      // Forces nodales / 节点受力
      m_fx = Kokkos::View<Real_t*>("fx", numNode);
      m_fy = Kokkos::View<Real_t*>("fy", numNode);
      m_fz = Kokkos::View<Real_t*>("fz", numNode);

      m_symmX = Kokkos::View<Index_t*>("symmX", numNode);
      m_symmY = Kokkos::View<Index_t*>("symmY", numNode);
      m_symmZ = Kokkos::View<Index_t*>("symmZ", numNode);

      // Masse nodale / 节点质量
      m_nodalMass = Kokkos::View<Real_t*>("nodalMass", numNode);
   }

   /*
    * Allocation des champs élémentaires persistants / 分配持久化单元字段
    */
   void AllocateElemPersistent(Int_t numElem)
   {
      // Connectivité élément → 8 nœuds / 单元到节点的 8 节点连接
      m_nodelist = Kokkos::View<Index_t*>("nodelist", 8 * numElem);

      // Connectivité par faces / 六个面方向连接
      m_lxim   = Kokkos::View<Index_t*>("lxim",   numElem);
      m_lxip   = Kokkos::View<Index_t*>("lxip",   numElem);
      m_letam  = Kokkos::View<Index_t*>("letam",  numElem);
      m_letap  = Kokkos::View<Index_t*>("letap",  numElem);
      m_lzetam = Kokkos::View<Index_t*>("lzetam", numElem);
      m_lzetap = Kokkos::View<Index_t*>("lzetap", numElem);

      // Conditions limites par face / 单元面边界条件
      m_elemBC = Kokkos::View<Int_t*>("elemBC", numElem);

      // Champs énergétiques et pression / 能量与压力等场
      m_e  = Kokkos::View<Real_t*>("e",  numElem);
      m_p  = Kokkos::View<Real_t*>("p",  numElem);

      // Termes de viscosité artificielle / 人工粘性项
      m_q  = Kokkos::View<Real_t*>("q",  numElem);
      m_ql = Kokkos::View<Real_t*>("ql", numElem);
      m_qq = Kokkos::View<Real_t*>("qq", numElem);

      // Volumes / 体积类字段
      m_v     = Kokkos::View<Real_t*>("v",     numElem);
      m_volo  = Kokkos::View<Real_t*>("volo",  numElem);
      m_delv  = Kokkos::View<Real_t*>("delv",  numElem);
      m_vdov  = Kokkos::View<Real_t*>("vdov",  numElem);
      m_vnew  = Kokkos::View<Real_t*>("vnew",  numElem);

      // Longueur caractéristique / 单元特征长度
      m_arealg = Kokkos::View<Real_t*>("arealg", numElem);

      // Vitesse du son / 声速
      m_ss = Kokkos::View<Real_t*>("ss", numElem);

      // Masse élémentaire / 单元质量
      m_elemMass = Kokkos::View<Real_t*>("elemMass", numElem);
   }

   /*
    * Allocation des gradients — temporaire / 分配梯度（临时字段）
    */
   void AllocateGradients(Int_t numElem, Int_t allElem)
   {
      // Gradients de position / 位置梯度
      m_delx_xi   = Kokkos::View<Real_t*>("delx_xi",   numElem);
      m_delx_eta  = Kokkos::View<Real_t*>("delx_eta",  numElem);
      m_delx_zeta = Kokkos::View<Real_t*>("delx_zeta", numElem);

      // Gradients de vitesse / 速度梯度
      m_delv_xi   = Kokkos::View<Real_t*>("delv_xi",   allElem);
      m_delv_eta  = Kokkos::View<Real_t*>("delv_eta",  allElem);
      m_delv_zeta = Kokkos::View<Real_t*>("delv_zeta", allElem);
   }

   /*
    * Libération des gradients / 释放梯度字段
    * Kokkos::View ne nécessite pas free — assignation à un View vide suffit.
    * Kokkos::View 不需要手动 free，赋空 View 即可释放。
    */
   void DeallocateGradients()
   {
      m_delx_xi   = Kokkos::View<Real_t*>();
      m_delx_eta  = Kokkos::View<Real_t*>();
      m_delx_zeta = Kokkos::View<Real_t*>();

      m_delv_xi   = Kokkos::View<Real_t*>();
      m_delv_eta  = Kokkos::View<Real_t*>();
      m_delv_zeta = Kokkos::View<Real_t*>();
   }

   /*
    * Allocation des déformations principales / 分配主应变
    */
   void AllocateStrains(Int_t numElem)
   {
      m_dxx = Kokkos::View<Real_t*>("dxx", numElem);
      m_dyy = Kokkos::View<Real_t*>("dyy", numElem);
      m_dzz = Kokkos::View<Real_t*>("dzz", numElem);
   }

   /*
    * Libération des déformations principales / 释放主应变 
    */
   void DeallocateStrains()
   {
      m_dxx = Kokkos::View<Real_t*>();
      m_dyy = Kokkos::View<Real_t*>();
      m_dzz = Kokkos::View<Real_t*>();
   }

// ---- View accessors (for Kokkos kernels) ----
   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> xdd_view() { return m_xdd; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> ydd_view() { return m_ydd; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> zdd_view() { return m_zdd; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> fx_view() { return m_fx; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> fy_view() { return m_fy; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> fz_view() { return m_fz; }

   KOKKOS_INLINE_FUNCTION
   Kokkos::View<Real_t*> nodalMass_view() { return m_nodalMass; }
 
	KOKKOS_INLINE_FUNCTION
	Real_t* fx_ptr()const { return m_fx.data(); }

	KOKKOS_INLINE_FUNCTION
	Real_t* fy_ptr()const { return m_fy.data(); }

	KOKKOS_INLINE_FUNCTION
	Real_t* fz_ptr()const { return m_fz.data(); }

 // ACCESSORS — Version compatible Kokkos::View
 // 访问器 — 保留引用语义（写），并提供 const 只读版本
 //

 //
 // Champs nodaux / 节点场
 //

 // Coordonnées nodales / 节点坐标
 KOKKOS_INLINE_FUNCTION
 Real_t& x(Index_t idx) { return m_x(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t x(Index_t idx) const { return m_x(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& y(Index_t idx) { return m_y(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t y(Index_t idx) const { return m_y(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& z(Index_t idx) { return m_z(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t z(Index_t idx) const { return m_z(idx); }

 // Vitesses nodales / 节点速度
 KOKKOS_INLINE_FUNCTION
 Real_t& xd(Index_t idx) { return m_xd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t xd(Index_t idx) const { return m_xd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& yd(Index_t idx) { return m_yd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t yd(Index_t idx) const { return m_yd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& zd(Index_t idx) { return m_zd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t zd(Index_t idx) const { return m_zd(idx); }

 // Accélérations nodales / 节点加速度
 KOKKOS_INLINE_FUNCTION
 Real_t& xdd(Index_t idx) { return m_xdd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t xdd(Index_t idx) const { return m_xdd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& ydd(Index_t idx) { return m_ydd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t ydd(Index_t idx) const { return m_ydd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& zdd(Index_t idx) { return m_zdd(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t zdd(Index_t idx) const { return m_zdd(idx); }

 // Forces nodales / 节点受力
 KOKKOS_INLINE_FUNCTION
 Real_t& fx(Index_t idx) { return m_fx(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t fx(Index_t idx) const { return m_fx(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& fy(Index_t idx) { return m_fy(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t fy(Index_t idx) const { return m_fy(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t& fz(Index_t idx) { return m_fz(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t fz(Index_t idx) const { return m_fz(idx); }

 // Masse nodale / 节点质量
 KOKKOS_INLINE_FUNCTION
 Real_t& nodalMass(Index_t idx) { return m_nodalMass(idx); }

 KOKKOS_INLINE_FUNCTION
 Real_t nodalMass(Index_t idx) const { return m_nodalMass(idx); }

 //
 // Plans de symétrie / 对称平面节点集（只读即可）
 //
 KOKKOS_INLINE_FUNCTION
 Index_t symmX(Index_t idx) const { return m_symmX(idx); }

 KOKKOS_INLINE_FUNCTION
 Index_t symmY(Index_t idx) const { return m_symmY(idx); }

 KOKKOS_INLINE_FUNCTION
 Index_t symmZ(Index_t idx) const { return m_symmZ(idx); }

 bool symmXempty() const { return m_symmX.extent(0) == 0; }
 bool symmYempty() const { return m_symmY.extent(0) == 0; }
 bool symmZempty() const { return m_symmZ.extent(0) == 0; }

   //
   // Champs élémentaires / 单元场
   //

   // Taille des régions / 区域大小（1D View）
   KOKKOS_INLINE_FUNCTION
   Index_t& regElemSize(Index_t idx) { return m_regElemSize(idx); }

   // Numéro de région par élément / 每个单元的区域编号
   KOKKOS_INLINE_FUNCTION
   Index_t& regNumList(Index_t idx) { return m_regNumList(idx); }

   // Pour compatibilité — renvoie pointeur / 返回指针兼容旧接口
   KOKKOS_INLINE_FUNCTION
   Index_t* regNumListPtr() { return m_regNumList.data(); }

   // Liste des éléments par région / 区域内单元列表（二位 View）
   KOKKOS_INLINE_FUNCTION
   Index_t& regElemlist(Index_t r, Index_t idx)
   {
      return m_regElemlist(r, idx);
   }

   // Connectivité élément-8-nœuds / 单元的 8 节点连接
   KOKKOS_INLINE_FUNCTION
   Index_t* nodelist(Index_t elem)
   {
      return &m_nodelist(8 * elem);
   }
   KOKKOS_INLINE_FUNCTION
   const Index_t* nodelist(Index_t elem) const 
   {
      return m_nodelist.data() + elem * this->numNode();
   }

   // Connectivité entre éléments / 单元间邻接关系
   KOKKOS_INLINE_FUNCTION Index_t& lxim(Index_t idx) { return m_lxim(idx); }
   KOKKOS_INLINE_FUNCTION Index_t& lxip(Index_t idx) { return m_lxip(idx); }
   KOKKOS_INLINE_FUNCTION Index_t& letam(Index_t idx){ return m_letam(idx); }
   KOKKOS_INLINE_FUNCTION Index_t& letap(Index_t idx){ return m_letap(idx); }
   KOKKOS_INLINE_FUNCTION Index_t& lzetam(Index_t idx){ return m_lzetam(idx); }
   KOKKOS_INLINE_FUNCTION Index_t& lzetap(Index_t idx){ return m_lzetap(idx); }

   // Conditions limites / 单元边界条件
   KOKKOS_INLINE_FUNCTION
   Int_t& elemBC(Index_t idx) { return m_elemBC(idx); }

   // Déformations principales / 主应变

// 写访问（用于更新）
KOKKOS_INLINE_FUNCTION
Real_t& dxx(Index_t idx) { return m_dxx(idx); }

KOKKOS_INLINE_FUNCTION
Real_t& dyy(Index_t idx) { return m_dyy(idx); }

KOKKOS_INLINE_FUNCTION
Real_t& dzz(Index_t idx) { return m_dzz(idx); }

// 读访问（const，用于计算）
KOKKOS_INLINE_FUNCTION
Real_t dxx(Index_t idx) const { return m_dxx(idx); }

KOKKOS_INLINE_FUNCTION
Real_t dyy(Index_t idx) const { return m_dyy(idx); }

KOKKOS_INLINE_FUNCTION
Real_t dzz(Index_t idx) const { return m_dzz(idx); }


   // Nouveau volume relatif / 新相对体积
   KOKKOS_INLINE_FUNCTION Real_t& vnew(Index_t idx)const { return m_vnew(idx); }

    // Gradients de vitesse / 速度梯度

// 写访问（非 const，用于更新）
KOKKOS_INLINE_FUNCTION
Real_t& delv_xi(Index_t idx) { return m_delv_xi(idx); }

KOKKOS_INLINE_FUNCTION
Real_t& delv_eta(Index_t idx) { return m_delv_eta(idx); }

KOKKOS_INLINE_FUNCTION
Real_t& delv_zeta(Index_t idx) { return m_delv_zeta(idx); }

// 读访问（const，用于计算）
KOKKOS_INLINE_FUNCTION
Real_t delv_xi(Index_t idx) const { return m_delv_xi(idx); }

KOKKOS_INLINE_FUNCTION
Real_t delv_eta(Index_t idx) const { return m_delv_eta(idx); }

KOKKOS_INLINE_FUNCTION
Real_t delv_zeta(Index_t idx) const { return m_delv_zeta(idx); }

  // Gradients de position / 位置梯度
   KOKKOS_INLINE_FUNCTION Real_t& delx_xi(Index_t idx){ return m_delx_xi(idx); }
   KOKKOS_INLINE_FUNCTION Real_t& delx_eta(Index_t idx){ return m_delx_eta(idx); }
   KOKKOS_INLINE_FUNCTION Real_t& delx_zeta(Index_t idx){ return m_delx_zeta(idx); }

   // Énergie / 能量
   KOKKOS_INLINE_FUNCTION Real_t& e(Index_t idx){ return m_e(idx); }
   KOKKOS_INLINE_FUNCTION const Real_t& p(Index_t idx) const { return m_p(idx); }

   // Pression / 压力
   KOKKOS_INLINE_FUNCTION Real_t& p(Index_t idx){ return m_p(idx); }

   // Viscosité artificielle / 人工粘性
   KOKKOS_INLINE_FUNCTION Real_t& q(Index_t idx){ return m_q(idx); }
   KOKKOS_INLINE_FUNCTION const Real_t& q(Index_t idx) const { return m_q(idx); }


   // Termes de viscosité lin/quadratiques / 粘性线性项、二次项
   KOKKOS_INLINE_FUNCTION Real_t& ql(Index_t idx){ return m_ql(idx); }
   KOKKOS_INLINE_FUNCTION Real_t& qq(Index_t idx){ return m_qq(idx); }

    // Volume relatif / 相对体积
KOKKOS_INLINE_FUNCTION
Real_t& v(Index_t idx) { return m_v(idx); }

KOKKOS_INLINE_FUNCTION
Real_t v(Index_t idx) const { return m_v(idx); }


// Variation de volume / 体积变化
KOKKOS_INLINE_FUNCTION
Real_t& delv(Index_t idx) { return m_delv(idx); }

KOKKOS_INLINE_FUNCTION
Real_t delv(Index_t idx) const { return m_delv(idx); }


// Volume de référence / 参考体积
KOKKOS_INLINE_FUNCTION
Real_t& volo(Index_t idx) { return m_volo(idx); }

KOKKOS_INLINE_FUNCTION
Real_t volo(Index_t idx) const { return m_volo(idx); }


// dérivée du volume / 体积导数
KOKKOS_INLINE_FUNCTION
Real_t& vdov(Index_t idx) { return m_vdov(idx); }

KOKKOS_INLINE_FUNCTION
Real_t vdov(Index_t idx) const { return m_vdov(idx); }


// Longueur caractéristique / 特征长度
KOKKOS_INLINE_FUNCTION
Real_t& arealg(Index_t idx) { return m_arealg(idx); }

KOKKOS_INLINE_FUNCTION
Real_t arealg(Index_t idx) const { return m_arealg(idx); }


// Vitesse du son / 声速
KOKKOS_INLINE_FUNCTION
Real_t& ss(Index_t idx) { return m_ss(idx); }

KOKKOS_INLINE_FUNCTION
Real_t ss(Index_t idx) const { return m_ss(idx); }


// Masse élémentaire / 单元质量
KOKKOS_INLINE_FUNCTION
Real_t& elemMass(Index_t idx) { return m_elemMass(idx); }

KOKKOS_INLINE_FUNCTION
Real_t elemMass(Index_t idx) const { return m_elemMass(idx); }


   // Informations de connectivité nodale / 节点邻接信息
   //

   // Nombre d'éléments connectés à un nœud / 一个节点关联的单元数量
   KOKKOS_INLINE_FUNCTION
   Index_t nodeElemCount(Index_t idx) const
   {
      return m_nodeElemStart(idx + 1) - m_nodeElemStart(idx);
   }

   // Liste des coins d'éléments connectés au nœud / 节点对应的所有单元角点索引
   KOKKOS_INLINE_FUNCTION
   Index_t* nodeElemCornerList(Index_t idx)
   {
      return &m_nodeElemCornerList(m_nodeElemStart(idx));
   }

   //
   // Paramètres physiques — coupe, densité, constantes du modèle
   // 物理参数访问器 — 切断阈值、密度、方程常数等
   //

   // Cutoffs / 截断阈值
   KOKKOS_INLINE_FUNCTION Real_t u_cut() const { return m_u_cut; }
   KOKKOS_INLINE_FUNCTION Real_t e_cut() const { return m_e_cut; }
   KOKKOS_INLINE_FUNCTION Real_t p_cut() const { return m_p_cut; }
   KOKKOS_INLINE_FUNCTION Real_t q_cut() const { return m_q_cut; }
   KOKKOS_INLINE_FUNCTION Real_t v_cut() const { return m_v_cut; }

   // Constantes numériques du modèle / 模型硬编码常数
   KOKKOS_INLINE_FUNCTION Real_t hgcoef() const { return m_hgcoef; }
   KOKKOS_INLINE_FUNCTION Real_t qstop() const { return m_qstop; }
   KOKKOS_INLINE_FUNCTION Real_t monoq_max_slope() const { return m_monoq_max_slope; }
   KOKKOS_INLINE_FUNCTION Real_t monoq_limiter_mult() const { return m_monoq_limiter_mult; }
   KOKKOS_INLINE_FUNCTION Real_t ss4o3() const { return m_ss4o3; }
   KOKKOS_INLINE_FUNCTION Real_t qlc_monoq() const { return m_qlc_monoq; }
   KOKKOS_INLINE_FUNCTION Real_t qqc_monoq() const { return m_qqc_monoq; }
   KOKKOS_INLINE_FUNCTION Real_t qqc() const { return m_qqc; }

   // Paramètres de l'EOS / 状态方程参数
   KOKKOS_INLINE_FUNCTION Real_t eosvmax() const { return m_eosvmax; }
   KOKKOS_INLINE_FUNCTION Real_t eosvmin() const { return m_eosvmin; }
   KOKKOS_INLINE_FUNCTION Real_t pmin() const { return m_pmin; }
   KOKKOS_INLINE_FUNCTION Real_t emin() const { return m_emin; }
   KOKKOS_INLINE_FUNCTION Real_t dvovmax() const { return m_dvovmax; }
   KOKKOS_INLINE_FUNCTION Real_t refdens() const { return m_refdens; }

   //
   // Paramètres temporels / 时间推进参数
   //

   KOKKOS_INLINE_FUNCTION Real_t& time() { return m_time; }
   KOKKOS_INLINE_FUNCTION Real_t& deltatime() { return m_deltatime; }
   KOKKOS_INLINE_FUNCTION Real_t& deltatimemultlb() { return m_deltatimemultlb; }
   KOKKOS_INLINE_FUNCTION Real_t& deltatimemultub() { return m_deltatimemultub; }
   KOKKOS_INLINE_FUNCTION Real_t& stoptime() { return m_stoptime; }
   KOKKOS_INLINE_FUNCTION Real_t& dtcourant() { return m_dtcourant; }
   KOKKOS_INLINE_FUNCTION Real_t& dthydro() { return m_dthydro; }
   KOKKOS_INLINE_FUNCTION Real_t& dtmax() { return m_dtmax; }
   KOKKOS_INLINE_FUNCTION Real_t& dtfixed() { return m_dtfixed; }

   KOKKOS_INLINE_FUNCTION Int_t& cycle() { return m_cycle; }

   //
   // Informations MPI / MPI 拓扑信息（本版本为单机 CPU，保持接口）
   //

   KOKKOS_INLINE_FUNCTION Index_t& numRanks() { return m_numRanks; }

   KOKKOS_INLINE_FUNCTION Index_t& colLoc() { return m_colLoc; }
   KOKKOS_INLINE_FUNCTION Index_t& rowLoc() { return m_rowLoc; }
   KOKKOS_INLINE_FUNCTION Index_t& planeLoc() { return m_planeLoc; }
   KOKKOS_INLINE_FUNCTION Index_t& tp() { return m_tp; }

   //
   // Informations géométriques du domaine / 网格空间信息
   //

   KOKKOS_INLINE_FUNCTION Index_t& sizeX() { return m_sizeX; }
   KOKKOS_INLINE_FUNCTION Index_t& sizeY() { return m_sizeY; }
   KOKKOS_INLINE_FUNCTION Index_t& sizeZ() { return m_sizeZ; }

   KOKKOS_INLINE_FUNCTION Index_t& numReg() { return m_numReg; }
   KOKKOS_INLINE_FUNCTION Int_t&  cost() { return m_cost; }

   KOKKOS_INLINE_FUNCTION Index_t& numElem() { return m_numElem; }
   KOKKOS_INLINE_FUNCTION Index_t numNode() const { return m_numNode; }

   KOKKOS_INLINE_FUNCTION Index_t& maxPlaneSize() { return m_maxPlaneSize; }
   KOKKOS_INLINE_FUNCTION Index_t& maxEdgeSize() { return m_maxEdgeSize; }

#if USE_MPI
   // Buffers MPI (non utilisés dans la version CPU) / MPI 缓冲区（CPU-only 不使用）
   Real_t* commDataSend;
   Real_t* commDataRecv;
   MPI_Request recvRequest[26];
   MPI_Request sendRequest[26];
#endif
private:

   /* 
    * BuildMesh et autres — fonctions internes utilisées pour construire 
    * les structures du domaine. / 构网格与初始化内部使用的私有方法 
    */
   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
   void SetupThreadSupportStructures();
   void CreateRegionIndexSets(Int_t nreg, Int_t balance);
   void SetupCommBuffers(Int_t edgeNodes);
   void SetupSymmetryPlanes(Int_t edgeNodes);
   void SetupElementConnectivities(Int_t edgeElems);
   void SetupBoundaryConditions(Int_t edgeElems);

   /*
    * Implémentation des données 
    * 以下为全部数据字段（由 vector/ptr 改为 Kokkos::View）
    */

   /* Node-centered fields / 以节点为中心的字段 */

   Kokkos::View<Real_t*> m_x;    /* coordinates / 坐标 */
   Kokkos::View<Real_t*> m_y;
   Kokkos::View<Real_t*> m_z;

   Kokkos::View<Real_t*> m_xd;   /* velocities / 速度 */
   Kokkos::View<Real_t*> m_yd;
   Kokkos::View<Real_t*> m_zd;

   Kokkos::View<Real_t*> m_xdd;  /* accelerations / 加速度 */
   Kokkos::View<Real_t*> m_ydd;
   Kokkos::View<Real_t*> m_zdd;

   Kokkos::View<Real_t*> m_fx;   /* forces / 力 */
   Kokkos::View<Real_t*> m_fy;
   Kokkos::View<Real_t*> m_fz;

   Kokkos::View<Real_t*> m_nodalMass; /* mass / 节点质量 */
   
   // ---- View accessors (for Kokkos kernels) ---- 

  /* Symmetry planes / 对称平面节点集 */
   Kokkos::View<Index_t*> m_symmX;
   Kokkos::View<Index_t*> m_symmY;
   Kokkos::View<Index_t*> m_symmZ;

   /* Element-centered fields / 以单元为中心的字段 */

   /* Region information / 区域信息 */
   Int_t m_numReg;
   Int_t m_cost;

   Kokkos::View<Index_t*> m_regElemSize;  /* Taille des régions / 每个区域包含的单元数 */
   Kokkos::View<Index_t*> m_regNumList;   /* Numéro de région par élément / 每个单元所属区域编号 */

   /* 
    * m_regElemlist : tableau 2D (numReg × taille_variable) 
    * 原始为 Index_t**，这里用二维 View 表示每个区域一个列表
    * 中文：每个区域都有一段单独的元素索引列表 
    */
   Kokkos::View<Index_t**> m_regElemlist;

   /* Connectivity / 单元连接关系 */
   Kokkos::View<Index_t*> m_nodelist; /* elemToNode connectivity / 单元到节点的映射 */

   Kokkos::View<Index_t*> m_lxim;   /* neighbors / 相邻单元 */
   Kokkos::View<Index_t*> m_lxip;
   Kokkos::View<Index_t*> m_letam;
   Kokkos::View<Index_t*> m_letap;
   Kokkos::View<Index_t*> m_lzetam;
   Kokkos::View<Index_t*> m_lzetap;

   Kokkos::View<Int_t*>   m_elemBC; /* boundary condition mask / 边界条件掩码 */

   /* Temporary strain tensors / 临时应变张量 */
   Kokkos::View<Real_t*> m_dxx;
   Kokkos::View<Real_t*> m_dyy;
   Kokkos::View<Real_t*> m_dzz;

   /* Gradient fields / 梯度场 */
   Kokkos::View<Real_t*> m_delv_xi;
   Kokkos::View<Real_t*> m_delv_eta;
   Kokkos::View<Real_t*> m_delv_zeta;

   Kokkos::View<Real_t*> m_delx_xi;
   Kokkos::View<Real_t*> m_delx_eta;
   Kokkos::View<Real_t*> m_delx_zeta;

   /* Physical quantities (per element) / 单元物理量 */
   Kokkos::View<Real_t*> m_e;      /* energy / 内能 */
   Kokkos::View<Real_t*> m_p;      /* pressure / 压力 */
   Kokkos::View<Real_t*> m_q;      /* artificial viscosity / 人工粘性 */
   Kokkos::View<Real_t*> m_ql;     /* linear term / 线性项 */
   Kokkos::View<Real_t*> m_qq;     /* quadratic term / 二次项 */

   Kokkos::View<Real_t*> m_v;      /* relative volume / 相对体积 */
   Kokkos::View<Real_t*> m_volo;   /* reference volume / 参考体积 */
   Kokkos::View<Real_t*> m_vnew;   /* new volume / 新体积 */
   Kokkos::View<Real_t*> m_delv;   /* vnew - v / 体积变化 */
   Kokkos::View<Real_t*> m_vdov;   /* volume derivative / 体积导数 */

   Kokkos::View<Real_t*> m_arealg; /* characteristic length / 特征长度 */

   Kokkos::View<Real_t*> m_ss;     /* sound speed / 声速 */

   Kokkos::View<Real_t*> m_elemMass; /* element mass / 单元质量 */

   /* Hard-coded model constants / 固定模型常数 */
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

   /* Simulation time parameters / 时间推进参数 */
   Real_t m_dtcourant;
   Real_t m_dthydro;
   Int_t  m_cycle;
   Real_t m_dtfixed;
   Real_t m_time;
   Real_t m_deltatime;
   Real_t m_deltatimemultlb;
   Real_t m_deltatimemultub;
   Real_t m_dtmax;
   Real_t m_stoptime;

   /* MPI info (placeholder) / MPI 信息（仅为接口保留） */
   Int_t m_numRanks;

   /* Domain geometric decomposition / 几何分解信息 */
   Index_t m_colLoc;
   Index_t m_rowLoc;
   Index_t m_planeLoc;
   Index_t m_tp;

   Index_t m_sizeX;
   Index_t m_sizeY;
   Index_t m_sizeZ;
   Index_t m_numElem;
   Index_t m_numNode;

   Index_t m_maxPlaneSize;
   Index_t m_maxEdgeSize;

   /* 
    * Thread support structures (converted to Views)
    * 线程支持结构（OpenMP hack，在 Kokkos 保留逻辑）
    */
   Kokkos::View<Index_t*> m_nodeElemStart;
   Kokkos::View<Index_t*> m_nodeElemCornerList;

   /* Bounds used during mesh construction / 网格构建时的边界信息 */
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax;
};

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

struct cmdLineOpts {
   Int_t its; // -i 
   Int_t nx;  // -s 
   Int_t numReg; // -r 
   Int_t numFiles; // -f
   Int_t showProg; // -p
   Int_t quiet; // -q
   Int_t viz; // -v 
   Int_t cost; // -c
   Int_t balance; // -b
};



// Function Prototypes

// lulesh-par
Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);

// lulesh-viz
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// lulesh-init
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);


#endif /* LULESH_KOKKOS_H */
