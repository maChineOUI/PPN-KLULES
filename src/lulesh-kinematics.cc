#include "lulesh-kinematics.h"
#include "lulesh-geometry.h"

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

void CalcLagrangeElements(Domain& domain, Kokkos::View<Real_t*> vnew)
{
   Index_t numElem = domain.numElem() ;
   if (numElem > 0) {
      const Real_t deltatime = domain.deltatime() ;

      domain.AllocateStrains(numElem);

      // Extract Views for KOKKOS_LAMBDA ([=] capture)
      auto nodelist_v = domain.m_conn.m_nodelist ;
      auto x_v        = domain.m_nodes.m_x ;
      auto y_v        = domain.m_nodes.m_y ;
      auto z_v        = domain.m_nodes.m_z ;
      auto xd_v       = domain.m_nodes.m_xd ;
      auto yd_v       = domain.m_nodes.m_yd ;
      auto zd_v       = domain.m_nodes.m_zd ;
      auto volo_v     = domain.m_elems.m_volo ;
      auto v_v        = domain.m_elems.m_v ;
      auto delv_v     = domain.m_elems.m_delv ;
      auto arealg_v   = domain.m_elems.m_arealg ;
      auto vdov_v     = domain.m_elems.m_vdov ;
      auto dxx_v      = domain.m_elems.m_dxx ;
      auto dyy_v      = domain.m_elems.m_dyy ;
      auto dzz_v      = domain.m_elems.m_dzz ;

      /* Opt-9: Fused kinematics + Lagrange */
      // P2-A: LaunchBounds<256,2> → compiler targets ≤128 regs/thread (was 144)
      //        → SM occupancy 25% → 50%+; no-op on CPU backends.
      using kin_policy_t = Kokkos::RangePolicy<Kokkos::LaunchBounds<256, 2>>;
      Kokkos::parallel_for("CalcKinematicsAndLagrange", kin_policy_t(0, numElem),
                           KOKKOS_LAMBDA(Index_t k) {
         Real_t B[3][8] ;
         Real_t D[6] ;
         Real_t x_local[8], y_local[8], z_local[8] ;
         Real_t xd_local[8], yd_local[8], zd_local[8] ;
         Real_t detJ = Real_t(0.0) ;

         // Inline CollectDomainNodesToElemNodes (cannot call host function on GPU)
         const Index_t* elemToNode = nodelist_v.data() + 8*k ;
         for (Index_t n = 0; n < 8; ++n) {
            Index_t gn = elemToNode[n] ;
            x_local[n] = x_v(gn) ;
            y_local[n] = y_v(gn) ;
            z_local[n] = z_v(gn) ;
         }

         Real_t volume = CalcElemVolume(x_local, y_local, z_local) ;
         Real_t relativeVolume = volume / volo_v(k) ;
         vnew(k) = relativeVolume ;
         delv_v(k) = relativeVolume - v_v(k) ;

         arealg_v(k) = CalcElemCharacteristicLength(x_local, y_local, z_local, volume) ;

         for (Index_t lnode = 0; lnode < 8; ++lnode) {
            Index_t gnode = elemToNode[lnode] ;
            xd_local[lnode] = xd_v(gnode) ;
            yd_local[lnode] = yd_v(gnode) ;
            zd_local[lnode] = zd_v(gnode) ;
         }

         Real_t dt2 = Real_t(0.5) * deltatime ;
         for (Index_t j = 0; j < 8; ++j) {
            x_local[j] -= dt2 * xd_local[j] ;
            y_local[j] -= dt2 * yd_local[j] ;
            z_local[j] -= dt2 * zd_local[j] ;
         }

         CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ) ;
         CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D) ;

         // Lagrange part: vdov + deviatoric strain
         Real_t vdov      = D[0] + D[1] + D[2] ;
         Real_t vdovthird = vdov / Real_t(3.0) ;
         vdov_v(k) = vdov ;
         dxx_v(k)  = D[0] - vdovthird ;
         dyy_v(k)  = D[1] - vdovthird ;
         dzz_v(k)  = D[2] - vdovthird ;

         if (vnew(k) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive relative volume in CalcKinematicsAndLagrange") ;
      });

      domain.DeallocateStrains();
   }
}
