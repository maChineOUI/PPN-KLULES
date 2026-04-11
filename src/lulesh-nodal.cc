#include "lulesh-nodal.h"
#include "lulesh-stress.h"
#include "lulesh-comm.h"
#include "lulesh-profile.h"
#include <cassert>

/******************************************/

static inline void CalcForceForNodes(Domain& domain)
{
  Index_t numNode = domain.numNode() ;

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
   assert(domain.sizeY() == size && domain.sizeZ() == size
          && "ApplyBC: symmetry plane nodesets assume a cubic domain") ;
   Index_t numNodeBC = (size+1)*(size+1) ;

   // P6: fuse three symmetry-plane BC kernels into one — two fewer kernel launches.
   // All three nodesets have the same size (numNodeBC = (size+1)^2) and the index
   // i means the same thing in each: position within the planar nodeset.
   {
      const bool doX = !domain.symmXempty() ;
      const bool doY = !domain.symmYempty() ;
      const bool doZ = !domain.symmZempty() ;
      auto symmX_v = domain.m_nodes.m_symmX ;
      auto symmY_v = domain.m_nodes.m_symmY ;
      auto symmZ_v = domain.m_nodes.m_symmZ ;
      auto xdd_v   = domain.m_nodes.m_xdd ;
      auto ydd_v   = domain.m_nodes.m_ydd ;
      auto zdd_v   = domain.m_nodes.m_zdd ;
      Kokkos::parallel_for("ApplyBC_XYZ", numNodeBC, KOKKOS_LAMBDA(Index_t i) {
         if (doX) xdd_v(symmX_v(i)) = Real_t(0.0) ;
         if (doY) ydd_v(symmY_v(i)) = Real_t(0.0) ;
         if (doZ) zdd_v(symmZ_v(i)) = Real_t(0.0) ;
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
   Real_t u_cut = domain.u_cut() ;

   CalcForceForNodes(domain);

#if USE_MPI
   // Halo exchange 1: accumulate node forces across subdomain boundaries.
   // GPU → host, then CommRecv/CommSend/CommSBN, then host → GPU.
   {
      Index_t EN = domain.sizeX() + 1 ;  // edgeNodes
      Index_t dx = EN, dy = EN, dz = EN ;

      // P3: reuse pre-allocated host mirrors — no per-step cudaMallocHost.
      auto& fx_h = g_comm.sbn_fx_h ;
      auto& fy_h = g_comm.sbn_fy_h ;
      auto& fz_h = g_comm.sbn_fz_h ;
      {
         ProfileScope sbnHostPullTimer(ProfileTimer::sbn_host_pull) ;
         Kokkos::deep_copy(fx_h, domain.m_nodes.m_fx) ;
         Kokkos::deep_copy(fy_h, domain.m_nodes.m_fy) ;
         Kokkos::deep_copy(fz_h, domain.m_nodes.m_fz) ;
      }

      std::vector<Real_t*> fields = { fx_h.data(), fy_h.data(), fz_h.data() } ;

      // Post receives before sending
      {
         ProfileScope sbnPostTimer(ProfileTimer::sbn_post, false) ;
         CommRecv(domain, MSG_COMM_SBN, 3, dx, dy, dz, true) ;
         CommSend(domain, MSG_COMM_SBN, 3, fields, dx, dy, dz, true) ;
      }

      // Wait + scatter-add received forces into local arrays
      {
         ProfileScope sbnWaitUnpackTimer(ProfileTimer::sbn_wait_unpack, false) ;
         CommSBN(domain, 3, fields) ;
      }

      // Write accumulated forces back to device
      {
         ProfileScope sbnHostPushTimer(ProfileTimer::sbn_host_push) ;
         Kokkos::deep_copy(domain.m_nodes.m_fx, fx_h) ;
         Kokkos::deep_copy(domain.m_nodes.m_fy, fy_h) ;
         Kokkos::deep_copy(domain.m_nodes.m_fz, fz_h) ;
      }

      CalcAccelerationForNodes(domain, domain.numNode()) ;
   }
#else
   CalcAccelerationForNodes(domain, domain.numNode());
#endif

   ApplyAccelerationBoundaryConditionsForNodes(domain);

   const Real_t delt = domain.deltatime() ;
   CalcVelocityAndPositionForNodes(domain, delt, u_cut, domain.numNode()) ;
}
