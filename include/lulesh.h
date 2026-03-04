#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <Kokkos_Core.hpp>


// Precision specification
using real4  = float ;
using real8  = double ;
using real10 = long double ;  // 10 bytes on x86

using Index_t = int ;   // array subscript and loop index
using Real_t  = real8 ; // floating point representation
using Int_t   = int ;   // integer representation

enum { VolumeError = -1, QStopError = -2 } ;

// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
constexpr Int_t XI_M        = 0x00007;
constexpr Int_t XI_M_SYMM   = 0x00001;
constexpr Int_t XI_M_FREE   = 0x00002;
constexpr Int_t XI_M_COMM   = 0x00004;

constexpr Int_t XI_P        = 0x00038;
constexpr Int_t XI_P_SYMM   = 0x00008;
constexpr Int_t XI_P_FREE   = 0x00010;
constexpr Int_t XI_P_COMM   = 0x00020;

constexpr Int_t ETA_M       = 0x001c0;
constexpr Int_t ETA_M_SYMM  = 0x00040;
constexpr Int_t ETA_M_FREE  = 0x00080;
constexpr Int_t ETA_M_COMM  = 0x00100;

constexpr Int_t ETA_P       = 0x00e00;
constexpr Int_t ETA_P_SYMM  = 0x00200;
constexpr Int_t ETA_P_FREE  = 0x00400;
constexpr Int_t ETA_P_COMM  = 0x00800;

constexpr Int_t ZETA_M      = 0x07000;
constexpr Int_t ZETA_M_SYMM = 0x01000;
constexpr Int_t ZETA_M_FREE = 0x02000;
constexpr Int_t ZETA_M_COMM = 0x04000;

constexpr Int_t ZETA_P      = 0x38000;
constexpr Int_t ZETA_P_SYMM = 0x08000;
constexpr Int_t ZETA_P_FREE = 0x10000;
constexpr Int_t ZETA_P_COMM = 0x20000;

//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

/*
 * The implementation of the data abstraction used for lulesh
 * resides entirely in the Domain class below.  You can change
 * grouping and interleaving of fields here to maximize data layout
 * efficiency for your underlying architecture or compiler.
 *
 * For example, fields can be implemented as STL objects or
 * raw array pointers.  As another example, individual fields
 * m_x, m_y, m_z could be budled into
 *
 *    struct { Real_t x, y, z ; } *m_coord ;
 *
 * allowing accessor functions such as
 *
 *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
 *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
 *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
 */

class Domain {

   public:

   // Constructor
   Domain(Int_t numRanks, Index_t colLoc,
          Index_t rowLoc, Index_t planeLoc,
          Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

   // Temporary field allocation (called from kinematics/viscosity modules)
   void AllocateGradients(Int_t numElem, Int_t allElem)
   {
      // Position gradients
      m_elems.m_delx_xi   = Kokkos::View<Real_t*>("delx_xi",   numElem);
      m_elems.m_delx_eta  = Kokkos::View<Real_t*>("delx_eta",  numElem);
      m_elems.m_delx_zeta = Kokkos::View<Real_t*>("delx_zeta", numElem);

      // Velocity gradients
      m_elems.m_delv_xi   = Kokkos::View<Real_t*>("delv_xi",   allElem);
      m_elems.m_delv_eta  = Kokkos::View<Real_t*>("delv_eta",  allElem);
      m_elems.m_delv_zeta = Kokkos::View<Real_t*>("delv_zeta", allElem);
   }

   void DeallocateGradients()
   {
      m_elems.m_delx_zeta = Kokkos::View<Real_t*>();
      m_elems.m_delx_eta  = Kokkos::View<Real_t*>();
      m_elems.m_delx_xi   = Kokkos::View<Real_t*>();

      m_elems.m_delv_zeta = Kokkos::View<Real_t*>();
      m_elems.m_delv_eta  = Kokkos::View<Real_t*>();
      m_elems.m_delv_xi   = Kokkos::View<Real_t*>();
   }

   void AllocateStrains(Int_t numElem)
   {
      m_elems.m_dxx = Kokkos::View<Real_t*>("dxx", numElem);
      m_elems.m_dyy = Kokkos::View<Real_t*>("dyy", numElem);
      m_elems.m_dzz = Kokkos::View<Real_t*>("dzz", numElem);
   }

   void DeallocateStrains()
   {
      m_elems.m_dzz = Kokkos::View<Real_t*>();
      m_elems.m_dyy = Kokkos::View<Real_t*>();
      m_elems.m_dxx = Kokkos::View<Real_t*>();
   }

   //
   // ACCESSORS
   //

   // Node-centered

   // Nodal coordinates
   Real_t& x(Index_t idx)    { return m_nodes.m_x[idx] ; }
   Real_t& y(Index_t idx)    { return m_nodes.m_y[idx] ; }
   Real_t& z(Index_t idx)    { return m_nodes.m_z[idx] ; }

   // Nodal velocities
   Real_t& xd(Index_t idx)   { return m_nodes.m_xd[idx] ; }
   Real_t& yd(Index_t idx)   { return m_nodes.m_yd[idx] ; }
   Real_t& zd(Index_t idx)   { return m_nodes.m_zd[idx] ; }

   // Nodal accelerations
   Real_t& xdd(Index_t idx)  { return m_nodes.m_xdd[idx] ; }
   Real_t& ydd(Index_t idx)  { return m_nodes.m_ydd[idx] ; }
   Real_t& zdd(Index_t idx)  { return m_nodes.m_zdd[idx] ; }

   // Nodal forces
   Real_t& fx(Index_t idx)   { return m_nodes.m_fx[idx] ; }
   Real_t& fy(Index_t idx)   { return m_nodes.m_fy[idx] ; }
   Real_t& fz(Index_t idx)   { return m_nodes.m_fz[idx] ; }

   // Nodal mass
   Real_t& nodalMass(Index_t idx) { return m_nodes.m_nodalMass[idx] ; }

   // Nodes on symmertry planes
   Index_t symmX(Index_t idx) { return m_nodes.m_symmX[idx] ; }
   Index_t symmY(Index_t idx) { return m_nodes.m_symmY[idx] ; }
   Index_t symmZ(Index_t idx) { return m_nodes.m_symmZ[idx] ; }
   bool symmXempty()          { return m_nodes.m_symmX.extent(0) == 0; }
   bool symmYempty()          { return m_nodes.m_symmY.extent(0) == 0; }
   bool symmZempty()          { return m_nodes.m_symmZ.extent(0) == 0; }

   //
   // Element-centered
   //
   Index_t&  regElemSize(Index_t idx) { return m_conn.m_regElemSize[idx] ; }
   Index_t&  regNumList(Index_t idx) { return m_conn.m_regNumList[idx] ; }
   Index_t*  regNumList()            { return m_conn.m_regNumList.data() ; }
   Index_t*  regElemlist(Int_t r)    { return m_conn.m_regElemlist[r].data() ; }
   Index_t&  regElemlist(Int_t r, Index_t idx) { return m_conn.m_regElemlist[r][idx] ; }

   Index_t*  nodelist(Index_t idx)    { return m_conn.m_nodelist.data() + Index_t(8)*idx ; }

   // elem connectivities through face
   Index_t&  lxim(Index_t idx) { return m_conn.m_lxim[idx] ; }
   Index_t&  lxip(Index_t idx) { return m_conn.m_lxip[idx] ; }
   Index_t&  letam(Index_t idx) { return m_conn.m_letam[idx] ; }
   Index_t&  letap(Index_t idx) { return m_conn.m_letap[idx] ; }
   Index_t&  lzetam(Index_t idx) { return m_conn.m_lzetam[idx] ; }
   Index_t&  lzetap(Index_t idx) { return m_conn.m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   Int_t&  elemBC(Index_t idx) { return m_conn.m_elemBC[idx] ; }

   // Principal strains - temporary
   Real_t& dxx(Index_t idx)  { return m_elems.m_dxx[idx] ; }
   Real_t& dyy(Index_t idx)  { return m_elems.m_dyy[idx] ; }
   Real_t& dzz(Index_t idx)  { return m_elems.m_dzz[idx] ; }

   // Velocity gradient - temporary
   Real_t& delv_xi(Index_t idx)    { return m_elems.m_delv_xi[idx] ; }
   Real_t& delv_eta(Index_t idx)   { return m_elems.m_delv_eta[idx] ; }
   Real_t& delv_zeta(Index_t idx)  { return m_elems.m_delv_zeta[idx] ; }

   // Position gradient - temporary
   Real_t& delx_xi(Index_t idx)    { return m_elems.m_delx_xi[idx] ; }
   Real_t& delx_eta(Index_t idx)   { return m_elems.m_delx_eta[idx] ; }
   Real_t& delx_zeta(Index_t idx)  { return m_elems.m_delx_zeta[idx] ; }

   // Energy
   Real_t& e(Index_t idx)          { return m_elems.m_e[idx] ; }

   // Pressure
   Real_t& p(Index_t idx)          { return m_elems.m_p[idx] ; }

   // Artificial viscosity
   Real_t& q(Index_t idx)          { return m_elems.m_q[idx] ; }

   // Linear term for q
   Real_t& ql(Index_t idx)         { return m_elems.m_ql[idx] ; }
   // Quadratic term for q
   Real_t& qq(Index_t idx)         { return m_elems.m_qq[idx] ; }

   // Relative volume
   Real_t& v(Index_t idx)          { return m_elems.m_v[idx] ; }
   Real_t& delv(Index_t idx)       { return m_elems.m_delv[idx] ; }

   // Reference volume
   Real_t& volo(Index_t idx)       { return m_elems.m_volo[idx] ; }

   // volume derivative over volume
   Real_t& vdov(Index_t idx)       { return m_elems.m_vdov[idx] ; }

   // Element characteristic length
   Real_t& arealg(Index_t idx)     { return m_elems.m_arealg[idx] ; }

   // Sound speed
   Real_t& ss(Index_t idx)         { return m_elems.m_ss[idx] ; }

   // Element mass
   Real_t& elemMass(Index_t idx)  { return m_elems.m_elemMass[idx] ; }

   Index_t nodeElemCount(Index_t idx)
   { return m_conn.m_nodeElemStart[idx+1] - m_conn.m_nodeElemStart[idx] ; }

   Index_t *nodeElemCornerList(Index_t idx)
   { return m_conn.m_nodeElemCornerList.data() + m_conn.m_nodeElemStart[idx] ; }

   // Parameters

   // Cutoffs
   Real_t u_cut() const               { return m_u_cut ; }
   Real_t e_cut() const               { return m_e_cut ; }
   Real_t p_cut() const               { return m_p_cut ; }
   Real_t q_cut() const               { return m_q_cut ; }
   Real_t v_cut() const               { return m_v_cut ; }

   // Other constants (usually are settable via input file in real codes)
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

   // Timestep controls, etc...
   Real_t& time()                 { return m_time ; }
   Real_t& deltatime()            { return m_deltatime ; }
   Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
   Real_t& deltatimemultub()      { return m_deltatimemultub ; }
   Real_t& stoptime()             { return m_stoptime ; }
   Real_t& dtcourant()            { return m_dtcourant ; }
   Real_t& dthydro()              { return m_dthydro ; }
   Real_t& dtmax()                { return m_dtmax ; }
   Real_t& dtfixed()              { return m_dtfixed ; }

   Int_t&  cycle()                { return m_cycle ; }
   Index_t&  numRanks()           { return m_numRanks ; }

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

   Index_t&  sizeX()              { return m_sizeX ; }
   Index_t&  sizeY()              { return m_sizeY ; }
   Index_t&  sizeZ()              { return m_sizeZ ; }
   Index_t&  numReg()             { return m_numReg ; }
   Int_t&  cost()             { return m_cost ; }
   Index_t&  numElem()            { return m_numElem ; }
   Index_t&  numNode()            { return m_numNode ; }

   private:

   //
   // ALLOCATION (called only from Domain constructor in lulesh-init.cc)
   //

   void AllocateNodePersistent(Int_t numNode) // Node-centered
   {
      m_nodes.m_x = Kokkos::View<Real_t*>("x", numNode);  // coordinates
      m_nodes.m_y = Kokkos::View<Real_t*>("y", numNode);
      m_nodes.m_z = Kokkos::View<Real_t*>("z", numNode);

      m_nodes.m_xd = Kokkos::View<Real_t*>("xd", numNode); // velocities
      m_nodes.m_yd = Kokkos::View<Real_t*>("yd", numNode);
      m_nodes.m_zd = Kokkos::View<Real_t*>("zd", numNode);

      m_nodes.m_xdd = Kokkos::View<Real_t*>("xdd", numNode); // accelerations
      m_nodes.m_ydd = Kokkos::View<Real_t*>("ydd", numNode);
      m_nodes.m_zdd = Kokkos::View<Real_t*>("zdd", numNode);

      m_nodes.m_fx = Kokkos::View<Real_t*>("fx", numNode);  // forces
      m_nodes.m_fy = Kokkos::View<Real_t*>("fy", numNode);
      m_nodes.m_fz = Kokkos::View<Real_t*>("fz", numNode);

      m_nodes.m_nodalMass = Kokkos::View<Real_t*>("nodalMass", numNode);  // mass
   }

   void AllocateElemPersistent(Int_t numElem) // Elem-centered
   {
      m_conn.m_nodelist = Kokkos::View<Index_t*>("nodelist", 8*numElem);

      // elem connectivities through face
      m_conn.m_lxim   = Kokkos::View<Index_t*>("lxim",   numElem);
      m_conn.m_lxip   = Kokkos::View<Index_t*>("lxip",   numElem);
      m_conn.m_letam  = Kokkos::View<Index_t*>("letam",  numElem);
      m_conn.m_letap  = Kokkos::View<Index_t*>("letap",  numElem);
      m_conn.m_lzetam = Kokkos::View<Index_t*>("lzetam", numElem);
      m_conn.m_lzetap = Kokkos::View<Index_t*>("lzetap", numElem);

      m_conn.m_elemBC = Kokkos::View<Int_t*>("elemBC", numElem);

      m_elems.m_e  = Kokkos::View<Real_t*>("e",  numElem);
      m_elems.m_p  = Kokkos::View<Real_t*>("p",  numElem);

      m_elems.m_q  = Kokkos::View<Real_t*>("q",  numElem);
      m_elems.m_ql = Kokkos::View<Real_t*>("ql", numElem);
      m_elems.m_qq = Kokkos::View<Real_t*>("qq", numElem);

      m_elems.m_v = Kokkos::View<Real_t*>("v", numElem);

      m_elems.m_volo  = Kokkos::View<Real_t*>("volo",  numElem);
      m_elems.m_delv  = Kokkos::View<Real_t*>("delv",  numElem);
      m_elems.m_vdov  = Kokkos::View<Real_t*>("vdov",  numElem);

      m_elems.m_arealg   = Kokkos::View<Real_t*>("arealg",   numElem);
      m_elems.m_ss       = Kokkos::View<Real_t*>("ss",       numElem);
      m_elems.m_elemMass = Kokkos::View<Real_t*>("elemMass", numElem);
   }

   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
   void SetupThreadSupportStructures();
   void CreateRegionIndexSets(Int_t nreg, Int_t balance);
   void SetupCommBuffers(Int_t edgeNodes);
   void SetupSymmetryPlanes(Int_t edgeNodes);
   void SetupElementConnectivities(Int_t edgeElems);
   void SetupBoundaryConditions(Int_t edgeElems);

   //
   // IMPLEMENTATION
   //

   /* Node-centered arrays */
   struct NodeArrays {
      Kokkos::View<Real_t*> m_x, m_y, m_z;        /* coordinates */
      Kokkos::View<Real_t*> m_xd, m_yd, m_zd;     /* velocities */
      Kokkos::View<Real_t*> m_xdd, m_ydd, m_zdd;  /* accelerations */
      Kokkos::View<Real_t*> m_fx, m_fy, m_fz;     /* forces */
      Kokkos::View<Real_t*> m_nodalMass;           /* mass */
      Kokkos::View<Index_t*> m_symmX, m_symmY, m_symmZ; /* symmetry plane nodesets */
   } m_nodes;

   /* Element connectivity */
   struct ElemConnectivity {
      Kokkos::View<Index_t*> m_nodelist;           /* elemToNode connectivity */
      Kokkos::View<Index_t*> m_lxim, m_lxip;      /* element connectivity across each face */
      Kokkos::View<Index_t*> m_letam, m_letap;
      Kokkos::View<Index_t*> m_lzetam, m_lzetap;
      Kokkos::View<Int_t*>   m_elemBC;             /* symmetry/free-surface flags for each elem face */
      Kokkos::View<Index_t*> m_regElemSize;        /* size of region sets */
      Kokkos::View<Index_t*> m_regNumList;         /* region number per domain element */
      std::vector<std::vector<Index_t>> m_regElemlist; /* region indexset (jagged — keep as vector) */
      Kokkos::View<Index_t*> m_nodeElemStart;      /* OMP: node-element index start */
      Kokkos::View<Index_t*> m_nodeElemCornerList; /* OMP: node-element corner list */
   } m_conn;

   /* Element state arrays */
   struct ElemState {
      Kokkos::View<Real_t*> m_dxx, m_dyy, m_dzz;  /* principal strains -- temporary */
      Kokkos::View<Real_t*> m_delv_xi, m_delv_eta, m_delv_zeta; /* velocity gradient -- temporary */
      Kokkos::View<Real_t*> m_delx_xi, m_delx_eta, m_delx_zeta; /* coordinate gradient -- temporary */
      Kokkos::View<Real_t*> m_e;                   /* energy */
      Kokkos::View<Real_t*> m_p;                   /* pressure */
      Kokkos::View<Real_t*> m_q, m_ql, m_qq;      /* q, linear term, quadratic term */
      Kokkos::View<Real_t*> m_v, m_volo, m_delv, m_vdov; /* relative/reference volume, derivatives */
      Kokkos::View<Real_t*> m_arealg;              /* characteristic length of an element */
      Kokkos::View<Real_t*> m_ss;                  /* "sound speed" */
      Kokkos::View<Real_t*> m_elemMass;            /* mass */
   } m_elems;

   // Region scalars
   Int_t    m_numReg ;
   Int_t    m_cost;                               /* imbalance cost */

   // Cutoffs (treat as constants)
   const Real_t  m_e_cut ;             // energy tolerance
   const Real_t  m_p_cut ;             // pressure tolerance
   const Real_t  m_q_cut ;             // q tolerance
   const Real_t  m_v_cut ;             // relative volume tolerance
   const Real_t  m_u_cut ;             // velocity tolerance

   // Other constants (usually setable, but hardcoded in this proxy app)

   const Real_t  m_hgcoef ;            // hourglass control
   const Real_t  m_ss4o3 ;
   const Real_t  m_qstop ;             // excessive q indicator
   const Real_t  m_monoq_max_slope ;
   const Real_t  m_monoq_limiter_mult ;
   const Real_t  m_qlc_monoq ;         // linear term coef for q
   const Real_t  m_qqc_monoq ;         // quadratic term coef for q
   const Real_t  m_qqc ;
   const Real_t  m_eosvmax ;
   const Real_t  m_eosvmin ;
   const Real_t  m_pmin ;              // pressure floor
   const Real_t  m_emin ;              // energy floor
   const Real_t  m_dvovmax ;           // maximum allowable volume change
   const Real_t  m_refdens ;           // reference density

   // Variables to keep track of timestep, simulation time, and cycle
   Real_t  m_dtcourant ;         // courant constraint
   Real_t  m_dthydro ;           // volume change constraint
   Int_t   m_cycle ;             // iteration count for simulation
   Real_t  m_dtfixed ;           // fixed time increment
   Real_t  m_time ;              // current time
   Real_t  m_deltatime ;         // variable time increment
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_dtmax ;             // maximum allowable time increment
   Real_t  m_stoptime ;          // end time for simulation


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

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

} ;


