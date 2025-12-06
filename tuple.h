#ifndef LULESH_TUPLE_H
#define LULESH_TUPLE_H

#if !defined(USE_MPI)
# error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

#if USE_MPI
#include <mpi.h>
#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#include <math.h>
#include <Kokkos_Core.hpp>
#include <vector>

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;

typedef int    Index_t ;
typedef real8  Real_t ;
typedef int    Int_t ;

enum { VolumeError = -1, QStopError = -2 } ;

inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline real4  FABS(real4  arg) { return fabsf(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real10 FABS(real10 arg) { return fabsl(arg) ; }

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

#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

struct cmdLineOpts {
   Int_t its;
   Int_t nx;
   Int_t numReg;
   Int_t numFiles;
   Int_t showProg;
   Int_t quiet;
   Int_t viz;
   Int_t cost;
   Int_t balance;
};

namespace lulesh_kokkos {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;
  template<class T> using View1D = Kokkos::View<T*, MemSpace>;
  template<class T> using View2D = Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>;
  template<class T> using HostView1D = Kokkos::View<T*, Kokkos::HostSpace>;
}

class Domain;

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

class Domain {

public:

  using ExecSpace = lulesh_kokkos::ExecSpace;
  using MemSpace  = lulesh_kokkos::MemSpace;
  template<class T> using View1D = lulesh_kokkos::View1D<T>;
  template<class T> using View2D = lulesh_kokkos::View2D<T>;
  template<class T> using HostView1D = lulesh_kokkos::HostView1D<T>;

  Domain(Int_t numRanks, Index_t colLoc,
         Index_t rowLoc, Index_t planeLoc,
         Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);
  ~Domain();

  void AllocateNodePersistent(Int_t numNode)
  {
     m_coord = View1D<Tuple3>("m_coord", numNode);
     m_vel   = View1D<Tuple3>("m_vel", numNode);
     m_acc   = View1D<Tuple3>("m_acc", numNode);
     m_force = View1D<Tuple3>("m_force", numNode);
     m_nodalMass = View1D<Real_t>("m_nodalMass", numNode);
  }

  void AllocateElemPersistent(Int_t numElem)
  {
     m_nodelist = View1D<Index_t>("m_nodelist", numElem*8);
     m_faceToElem = View1D<FaceElemConn>("m_faceToElem", numElem);
     m_elemBC = View1D<Int_t>("m_elemBC", numElem);
     m_e = View1D<Real_t>("m_e", numElem);
     m_pq = View1D<Pcomponents>("m_pq", numElem);
     m_qlqq = View1D<Qcomponents>("m_qlqq", numElem);
     m_vol = View1D<Volume>("m_vol", numElem);
     m_vnew = View1D<Real_t>("m_vnew", numElem);
     m_delv = View1D<Real_t>("m_delv", numElem);
     m_vdov = View1D<Real_t>("m_vdov", numElem);
     m_arealg = View1D<Real_t>("m_arealg", numElem);
     m_ss = View1D<Real_t>("m_ss", numElem);
     m_elemMass = View1D<Real_t>("m_elemMass", numElem);
  }

  void AllocateGradients(Int_t numElem, Int_t allElem)
  {
     m_delx_xi   = View1D<Real_t>("m_delx_xi", numElem);
     m_delx_eta  = View1D<Real_t>("m_delx_eta", numElem);
     m_delx_zeta = View1D<Real_t>("m_delx_zeta", numElem);
     m_delv_xi   = View1D<Real_t>("m_delv_xi", allElem);
     m_delv_eta  = View1D<Real_t>("m_delv_eta", allElem);
     m_delv_zeta = View1D<Real_t>("m_delv_zeta", allElem);
  }

  void DeallocateGradients() { }

  void AllocateStrains(Int_t numElem)
  {
     m_dxx = View1D<Real_t>("m_dxx", numElem);
     m_dyy = View1D<Real_t>("m_dyy", numElem);
     m_dzz = View1D<Real_t>("m_dzz", numElem);
  }

  void DeallocateStrains() { }

  struct Tuple3 { Real_t x, y, z; };
  struct FaceElemConn { Index_t lxim, lxip, letam, letap, lzetam, lzetap; };
  struct Pcomponents { Real_t p, q; };
  struct Qcomponents { Real_t ql, qq; };
  struct Volume { Real_t v, volo; };

  KOKKOS_INLINE_FUNCTION Real_t& x(Index_t idx)    { return m_coord(idx).x ; }
  KOKKOS_INLINE_FUNCTION Real_t& y(Index_t idx)    { return m_coord(idx).y ; }
  KOKKOS_INLINE_FUNCTION Real_t& z(Index_t idx)    { return m_coord(idx).z ; }

  KOKKOS_INLINE_FUNCTION Real_t& xd(Index_t idx)   { return m_vel(idx).x ; }
  KOKKOS_INLINE_FUNCTION Real_t& yd(Index_t idx)   { return m_vel(idx).y ; }
  KOKKOS_INLINE_FUNCTION Real_t& zd(Index_t idx)   { return m_vel(idx).z ; }

  KOKKOS_INLINE_FUNCTION Real_t& xdd(Index_t idx)  { return m_acc(idx).x ; }
  KOKKOS_INLINE_FUNCTION Real_t& ydd(Index_t idx)  { return m_acc(idx).y ; }
  KOKKOS_INLINE_FUNCTION Real_t& zdd(Index_t idx)  { return m_acc(idx).z ; }

  KOKKOS_INLINE_FUNCTION Real_t& fx(Index_t idx)   { return m_force(idx).x ; }
  KOKKOS_INLINE_FUNCTION Real_t& fy(Index_t idx)   { return m_force(idx).y ; }
  KOKKOS_INLINE_FUNCTION Real_t& fz(Index_t idx)   { return m_force(idx).z ; }

  KOKKOS_INLINE_FUNCTION Real_t& nodalMass(Index_t idx) { return m_nodalMass(idx) ; }

  Index_t symmX(Index_t idx) const { return m_symmX_host.at(idx) ; }
  Index_t symmY(Index_t idx) const { return m_symmY_host.at(idx) ; }
  Index_t symmZ(Index_t idx) const { return m_symmZ_host.at(idx) ; }
  bool symmXempty() const { return m_symmX_host.empty(); }
  bool symmYempty() const { return m_symmY_host.empty(); }
  bool symmZempty() const { return m_symmZ_host.empty(); }

  KOKKOS_INLINE_FUNCTION Index_t&  regElemSize(Index_t idx) { return m_regElemSize(idx) ; }
  KOKKOS_INLINE_FUNCTION Index_t&  regNumList(Index_t idx) { return m_regNumList(idx) ; }
  Index_t* regNumList_ptr_host() { return m_regNumList_host.data(); }
  Index_t* regElemlist_ptr_host(Int_t r) { return m_regElemlist_host[r].data(); }
  Index_t& regElemlist_at(Int_t r, Index_t idx) { return m_regElemlist_host[r][idx] ; }

  KOKKOS_INLINE_FUNCTION Index_t nodelist_at(Index_t elem, int corner) const { return m_nodelist(elem*8 + corner); }

  KOKKOS_INLINE_FUNCTION Index_t&  lxim(Index_t idx) { return m_faceToElem(idx).lxim ; }
  KOKKOS_INLINE_FUNCTION Index_t&  lxip(Index_t idx) { return m_faceToElem(idx).lxip ; }
  KOKKOS_INLINE_FUNCTION Index_t&  letam(Index_t idx) { return m_faceToElem(idx).letam ; }
  KOKKOS_INLINE_FUNCTION Index_t&  letap(Index_t idx) { return m_faceToElem(idx).letap ; }
  KOKKOS_INLINE_FUNCTION Index_t&  lzetam(Index_t idx) { return m_faceToElem(idx).lzetam ; }
  KOKKOS_INLINE_FUNCTION Index_t&  lzetap(Index_t idx) { return m_faceToElem(idx).lzetap ; }

  KOKKOS_INLINE_FUNCTION Int_t&  elemBC(Index_t idx) { return m_elemBC(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& dxx(Index_t idx)  { return m_dxx(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& dyy(Index_t idx)  { return m_dyy(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& dzz(Index_t idx)  { return m_dzz(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& vnew(Index_t idx)  { return m_vnew(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& delv_xi(Index_t idx)    { return m_delv_xi(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& delv_eta(Index_t idx)   { return m_delv_eta(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& delx_xi(Index_t idx)    { return m_delx_xi(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& delx_eta(Index_t idx)   { return m_delx_eta(idx) ; }
  KOKKOS_INLINE_FUNCTION Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& e(Index_t idx)          { return m_e(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& p(Index_t idx)          { return m_pq(idx).p ; }

  KOKKOS_INLINE_FUNCTION Real_t& q(Index_t idx)          { return m_pq(idx).q ; }

  KOKKOS_INLINE_FUNCTION Real_t& ql(Index_t idx)         { return m_qlqq(idx).ql ; }
  KOKKOS_INLINE_FUNCTION Real_t& qq(Index_t idx)         { return m_qlqq(idx).qq ; }

  KOKKOS_INLINE_FUNCTION Real_t& delv(Index_t idx)       { return m_delv(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& v(Index_t idx)          { return m_vol(idx).v ; }
  KOKKOS_INLINE_FUNCTION Real_t& volo(Index_t idx)       { return m_vol(idx).volo ; }

  KOKKOS_INLINE_FUNCTION Real_t& vdov(Index_t idx)       { return m_vdov(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& arealg(Index_t idx)     { return m_arealg(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& ss(Index_t idx)         { return m_ss(idx) ; }

  KOKKOS_INLINE_FUNCTION Real_t& elemMass(Index_t idx)  { return m_elemMass(idx) ; }

  KOKKOS_INLINE_FUNCTION Index_t nodeElemCount(Index_t idx) const
  { return static_cast<Index_t>(m_nodeElemStart_host_vector[idx+1] - m_nodeElemStart_host_vector[idx]); }

  Index_t *nodeElemCornerList(Index_t idx)
  { return m_nodeElemCornerList_host_vector.data() + m_nodeElemStart_host_vector[idx]; }

  Real_t u_cut() const               { return m_u_cut ; }
  Real_t e_cut() const               { return m_e_cut ; }
  Real_t p_cut() const               { return m_p_cut ; }
  Real_t q_cut() const               { return m_q_cut ; }
  Real_t v_cut() const               { return m_v_cut ; }

  Real_t hgcoef() const              { return m_hgcoef ; }
  Real_t qstop() const               { return m_qstop ; }
  Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
  Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
  Real_t ss4o3() const               { return m_ss4o3 ; }
  Real_t qlc_monoq() const           { return m_qlc_monoq ; }
  Real_t qqc_monoq() const           { return m_qqc_monoq ; }
  Real_t qqc() const                 { return m_qqc ; }

  Real_t eosvmax() const             { return m_eosvmax ; }
  Real_t eosvmin() const             { return m_eosvmin ; }
  Real_t pmin() const                { return m_pmin ; }
  Real_t emin() const                { return m_emin ; }
  Real_t dvovmax() const             { return m_dvovmax ; }
  Real_t refdens() const             { return m_refdens ; }

  KOKKOS_INLINE_FUNCTION Real_t& time()                 { return m_time ; }
  KOKKOS_INLINE_FUNCTION Real_t& deltatime()            { return m_deltatime ; }
  KOKKOS_INLINE_FUNCTION Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
  KOKKOS_INLINE_FUNCTION Real_t& deltatimemultub()      { return m_deltatimemultub ; }
  KOKKOS_INLINE_FUNCTION Real_t& stoptime()             { return m_stoptime ; }
  KOKKOS_INLINE_FUNCTION Real_t& dtcourant()            { return m_dtcourant ; }
  KOKKOS_INLINE_FUNCTION Real_t& dthydro()              { return m_dthydro ; }
  KOKKOS_INLINE_FUNCTION Real_t& dtmax()                { return m_dtmax ; }
  KOKKOS_INLINE_FUNCTION Real_t& dtfixed()              { return m_dtfixed ; }

  KOKKOS_INLINE_FUNCTION Int_t&  cycle()                { return m_cycle ; }
  KOKKOS_INLINE_FUNCTION Index_t&  numRanks()           { return m_numRanks ; }

  KOKKOS_INLINE_FUNCTION Index_t&  colLoc()             { return m_colLoc ; }
  KOKKOS_INLINE_FUNCTION Index_t&  rowLoc()             { return m_rowLoc ; }
  KOKKOS_INLINE_FUNCTION Index_t&  planeLoc()           { return m_planeLoc ; }
  KOKKOS_INLINE_FUNCTION Index_t&  tp()                 { return m_tp ; }

  KOKKOS_INLINE_FUNCTION Index_t&  sizeX()              { return m_sizeX ; }
  KOKKOS_INLINE_FUNCTION Index_t&  sizeY()              { return m_sizeY ; }
  KOKKOS_INLINE_FUNCTION Index_t&  sizeZ()              { return m_sizeZ ; }
  KOKKOS_INLINE_FUNCTION Index_t&  numReg()             { return m_numReg ; }
  KOKKOS_INLINE_FUNCTION Int_t&  cost()                  { return m_cost ; }
  KOKKOS_INLINE_FUNCTION Index_t&  numElem()            { return m_numElem ; }
  KOKKOS_INLINE_FUNCTION Index_t&  numNode()            { return m_numNode ; }

  KOKKOS_INLINE_FUNCTION Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
  KOKKOS_INLINE_FUNCTION Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }

#if USE_MPI
  HostView1D<Real_t> commDataSend ;
  HostView1D<Real_t> commDataRecv ;
  MPI_Request recvRequest[26] ;
  MPI_Request sendRequest[26] ;
#endif

private:

  void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(Int_t nreg, Int_t balance);
  void SetupCommBuffers(Int_t edgeNodes);
  void SetupSymmetryPlanes(Int_t edgeNodes);
  void SetupElementConnectivities(Int_t edgeElems);
  void SetupBoundaryConditions(Int_t edgeElems);

  View1D<Tuple3> m_coord ;
  View1D<Tuple3> m_vel ;
  View1D<Tuple3> m_acc ;
  View1D<Tuple3> m_force ;
  View1D<Real_t> m_nodalMass ;

  std::vector<Index_t> m_symmX_host ;
  std::vector<Index_t> m_symmY_host ;
  std::vector<Index_t> m_symmZ_host ;

  Int_t    m_numReg ;
  Int_t    m_cost;
  View1D<Index_t> m_regElemSize ;
  View1D<Index_t> m_regNumList ;
  std::vector<std::vector<Index_t>> m_regElemlist_host ;

  View1D<Index_t> m_nodelist ;
  View1D<FaceElemConn> m_faceToElem ;
  View1D<Int_t> m_elemBC ;
  View1D<Real_t> m_dxx ;
  View1D<Real_t> m_dyy ;
  View1D<Real_t> m_dzz ;
  View1D<Real_t> m_delv_xi ;
  View1D<Real_t> m_delv_eta ;
  View1D<Real_t> m_delv_zeta ;
  View1D<Real_t> m_delx_xi ;
  View1D<Real_t> m_delx_eta ;
  View1D<Real_t> m_delx_zeta ;
  View1D<Real_t> m_e ;
  View1D<Pcomponents> m_pq ;
  View1D<Qcomponents> m_qlqq ;
  View1D<Volume> m_vol ;
  View1D<Real_t> m_vnew ;
  View1D<Real_t> m_delv ;
  View1D<Real_t> m_vdov ;
  View1D<Real_t> m_arealg ;
  View1D<Real_t> m_ss ;
  View1D<Real_t> m_elemMass ;

  const Real_t  m_e_cut ;
  const Real_t  m_p_cut ;
  const Real_t  m_q_cut ;
  const Real_t  m_v_cut ;
  const Real_t  m_u_cut ;

  const Real_t  m_hgcoef ;
  const Real_t  m_ss4o3 ;
  const Real_t  m_qstop ;
  const Real_t  m_monoq_max_slope ;
  const Real_t  m_monoq_limiter_mult ;
  const Real_t  m_qlc_monoq ;
  const Real_t  m_qqc_monoq ;
  const Real_t  m_qqc ;
  const Real_t  m_eosvmax ;
  const Real_t  m_eosvmin ;
  const Real_t  m_pmin ;
  const Real_t  m_emin ;
  const Real_t  m_dvovmax ;
  const Real_t  m_refdens ;

  Real_t  m_dtcourant ;
  Real_t  m_dthydro ;
  Int_t   m_cycle ;
  Real_t  m_dtfixed ;
  Real_t  m_time ;
  Real_t  m_deltatime ;
  Real_t  m_deltatimemultlb ;
  Real_t  m_deltatimemultub ;
  Real_t  m_dtmax ;
  Real_t  m_stoptime ;

  Int_t   m_numRanks ;

  Index_t m_colLoc ;
  Index_t m_rowLoc ;
  Index_t m_planeLoc ;
  Index_t m_tp ;

  Index_t m_sizeX ;
  Index_t m_sizeY ;
  Index_t m_sizeZ ;
  Index_t m_numElem ;
  Index_t m_numNode ;

  Index_t m_maxPlaneSize ;
  Index_t m_maxEdgeSize ;

  HostView1D<Index_t> m_nodeElemStart_host_view ;
  HostView1D<Index_t> m_nodeElemCornerList_host_view ;
  std::vector<Index_t> m_nodeElemCornerList_host_vector ;
  std::vector<Index_t> m_nodeElemStart_host_vector ;

  Index_t m_rowMin, m_rowMax;
  Index_t m_colMin, m_colMax;
  Index_t m_planeMin, m_planeMax ;

  std::vector<int> m_regElemSize_host ;
  std::vector<std::vector<Index_t>> m_regElemlist_host_full ;

};

Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]);

void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);

void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

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

void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);

#endif
