#include "lulesh-nodal.h"
#include "lulesh-comm.h"
#include "lulesh-stress.h"

/******************************************/

static inline void CalcForceForNodes(Domain& domain)
{
  Index_t numNode = domain.numNode() ;

#if USE_MPI
  CommRecv(domain, MSG_COMM_SBN, 3,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
           true, false) ;
#endif

  auto fx_v = domain.m_nodes.m_fx ;
  auto fy_v = domain.m_nodes.m_fy ;
  auto fz_v = domain.m_nodes.m_fz ;

  Kokkos::parallel_for("CalcForceForNodes", numNode, KOKKOS_LAMBDA(Index_t i) {
     fx_v(i) = Real_t(0.0) ;
     fy_v(i) = Real_t(0.0) ;
     fz_v(i) = Real_t(0.0) ;
  });

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

#if USE_MPI
  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::fx ;
  fieldData[1] = &Domain::fy ;
  fieldData[2] = &Domain::fz ;

  CommSend(domain, MSG_COMM_SBN, 3, fieldData,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
           true, false) ;
  CommSBN(domain, 3, fieldData) ;
#endif
}

/******************************************/

static inline
void CalcAccelerationForNodes(Domain& domain, Index_t numNode)
{
  auto fx_v        = domain.m_nodes.m_fx ;
  auto fy_v        = domain.m_nodes.m_fy ;
  auto fz_v        = domain.m_nodes.m_fz ;
  auto xdd_v       = domain.m_nodes.m_xdd ;
  auto ydd_v       = domain.m_nodes.m_ydd ;
  auto zdd_v       = domain.m_nodes.m_zdd ;
  auto nodalMass_v = domain.m_nodes.m_nodalMass ;

  Kokkos::parallel_for("CalcAccelerationForNodes", numNode, KOKKOS_LAMBDA(Index_t i) {
     xdd_v(i) = fx_v(i) / nodalMass_v(i) ;
     ydd_v(i) = fy_v(i) / nodalMass_v(i) ;
     zdd_v(i) = fz_v(i) / nodalMass_v(i) ;
  });
}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
{
   Index_t size = domain.sizeX();
   Index_t numNodeBC = (size+1)*(size+1) ;

   if (!domain.symmXempty()) {
      auto symmX_v = domain.m_nodes.m_symmX ;
      auto xdd_v   = domain.m_nodes.m_xdd ;
      Kokkos::parallel_for("ApplyBC_X", numNodeBC, KOKKOS_LAMBDA(Index_t i) {
         xdd_v(symmX_v(i)) = Real_t(0.0) ;
      });
   }
   if (!domain.symmYempty()) {
      auto symmY_v = domain.m_nodes.m_symmY ;
      auto ydd_v   = domain.m_nodes.m_ydd ;
      Kokkos::parallel_for("ApplyBC_Y", numNodeBC, KOKKOS_LAMBDA(Index_t i) {
         ydd_v(symmY_v(i)) = Real_t(0.0) ;
      });
   }
   if (!domain.symmZempty()) {
      auto symmZ_v = domain.m_nodes.m_symmZ ;
      auto zdd_v   = domain.m_nodes.m_zdd ;
      Kokkos::parallel_for("ApplyBC_Z", numNodeBC, KOKKOS_LAMBDA(Index_t i) {
         zdd_v(symmZ_v(i)) = Real_t(0.0) ;
      });
   }
}

/******************************************/

static inline
void CalcVelocityAndPositionForNodes(Domain& domain, const Real_t dt,
                                     const Real_t u_cut, Index_t numNode)
{
  auto xd_v  = domain.m_nodes.m_xd ;
  auto yd_v  = domain.m_nodes.m_yd ;
  auto zd_v  = domain.m_nodes.m_zd ;
  auto xdd_v = domain.m_nodes.m_xdd ;
  auto ydd_v = domain.m_nodes.m_ydd ;
  auto zdd_v = domain.m_nodes.m_zdd ;
  auto x_v   = domain.m_nodes.m_x ;
  auto y_v   = domain.m_nodes.m_y ;
  auto z_v   = domain.m_nodes.m_z ;

  /* Fused: velocity update then position update per node. */
  Kokkos::parallel_for("CalcVelocityAndPositionForNodes", numNode,
                       KOKKOS_LAMBDA(Index_t i) {
     Real_t xdtmp = xd_v(i) + xdd_v(i) * dt ;
     if (Kokkos::fabs(xdtmp) < u_cut) xdtmp = Real_t(0.0) ;
     xd_v(i) = xdtmp ;

     Real_t ydtmp = yd_v(i) + ydd_v(i) * dt ;
     if (Kokkos::fabs(ydtmp) < u_cut) ydtmp = Real_t(0.0) ;
     yd_v(i) = ydtmp ;

     Real_t zdtmp = zd_v(i) + zdd_v(i) * dt ;
     if (Kokkos::fabs(zdtmp) < u_cut) zdtmp = Real_t(0.0) ;
     zd_v(i) = zdtmp ;

     x_v(i) += xdtmp * dt ;
     y_v(i) += ydtmp * dt ;
     z_v(i) += zdtmp * dt ;
  });
}

/******************************************/

void LagrangeNodal(Domain& domain)
{
#if USE_MPI
   Domain_member fieldData[6] ;
#endif
   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

  CalcForceForNodes(domain);

#if USE_MPI
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
#endif

   CalcAccelerationForNodes(domain, domain.numNode());

   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityAndPositionForNodes(domain, delt, u_cut, domain.numNode()) ;

#if USE_MPI
   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
   CommSyncPosVel(domain) ;
#endif

  return;
}
