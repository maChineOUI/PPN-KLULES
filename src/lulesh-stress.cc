#include <cstdlib>
#include <Kokkos_ScatterView.hpp>
#include "lulesh-stress.h"
#include "lulesh-geometry.h"

/******************************************/

KOKKOS_INLINE_FUNCTION
static void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t fx[], Real_t fy[], Real_t fz[] )
{
   for (Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i] );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

/* P2-B: ScatterView replaces scatter-buffer + gather-kernel pattern.
   Node forces (fx/fy/fz) are zeroed by CalcForceForNodes immediately before
   this call, so ScatterView contribute (sum) is equivalent to the previous
   gather assignment.  Eliminates fx_elem/fy_elem/fz_elem scratch buffers and
   the separate gather parallel_for (saves 1 kernel launch + 1 barrier). */
static inline
void IntegrateStressForElems( Domain& domain,
                              Kokkos::View<Real_t*> determ,
                              Index_t numElem)
{
   // Extract Views for KOKKOS_LAMBDA
   auto nodelist_v = domain.m_conn.m_nodelist ;
   auto x_v        = domain.m_nodes.m_x ;
   auto y_v        = domain.m_nodes.m_y ;
   auto z_v        = domain.m_nodes.m_z ;
   auto p_v        = domain.m_elems.m_p ;
   auto q_v        = domain.m_elems.m_q ;
   auto fx_v       = domain.m_nodes.m_fx ;
   auto fy_v       = domain.m_nodes.m_fy ;
   auto fz_v       = domain.m_nodes.m_fz ;

   auto fx_scatter = Kokkos::Experimental::create_scatter_view(fx_v) ;
   auto fy_scatter = Kokkos::Experimental::create_scatter_view(fy_v) ;
   auto fz_scatter = Kokkos::Experimental::create_scatter_view(fz_v) ;

   Kokkos::parallel_for("IntegrateStressForElems", numElem,
                        KOKKOS_LAMBDA(Index_t k) {
      const Index_t* elemToNode = nodelist_v.data() + 8*k ;
      Real_t B[3][8] ;
      Real_t x_local[8], y_local[8], z_local[8] ;
      Real_t detJ_k ;

      // Inline CollectDomainNodesToElemNodes
      for (Index_t n = 0; n < 8; ++n) {
         Index_t gn = elemToNode[n] ;
         x_local[n] = x_v(gn) ;
         y_local[n] = y_v(gn) ;
         z_local[n] = z_v(gn) ;
      }

      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ_k) ;
      determ(k) = detJ_k ;

      CalcElemNodeNormals(B[0], B[1], B[2], x_local, y_local, z_local) ;

      /* Opt-8: inline InitStressTermsForElems */
      Real_t sig = -p_v(k) - q_v(k) ;

      Real_t fx_local[8], fy_local[8], fz_local[8] ;
      SumElemStressesToNodeForces(B, sig, sig, sig, fx_local, fy_local, fz_local) ;

      /* P2-B: scatter directly to node forces via atomicAdd (CUDA) */
      auto fx_acc = fx_scatter.access() ;
      auto fy_acc = fy_scatter.access() ;
      auto fz_acc = fz_scatter.access() ;
      for (Index_t ni = 0; ni < 8; ++ni) {
         fx_acc(elemToNode[ni]) += fx_local[ni] ;
         fy_acc(elemToNode[ni]) += fy_local[ni] ;
         fz_acc(elemToNode[ni]) += fz_local[ni] ;
      }
   });

   Kokkos::Experimental::contribute(fx_v, fx_scatter) ;
   Kokkos::Experimental::contribute(fy_v, fy_scatter) ;
   Kokkos::Experimental::contribute(fz_v, fz_scatter) ;
}

/******************************************/

KOKKOS_INLINE_FUNCTION
static void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                              Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz)
{
   Real_t hxx[4];
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

static inline
void CalcHourglassControlForElems(Domain& domain,
                                  Kokkos::View<Real_t*> determ, Real_t hgcoef)
{
   Index_t numElem  = domain.numElem() ;

   // Hourglass gamma matrix (Flanagan-Belytschko)
   const Real_t gamma[4][8] = {
      { Real_t( 1.), Real_t( 1.), Real_t(-1.), Real_t(-1.),
        Real_t(-1.), Real_t(-1.), Real_t( 1.), Real_t( 1.) },
      { Real_t( 1.), Real_t(-1.), Real_t(-1.), Real_t( 1.),
        Real_t(-1.), Real_t( 1.), Real_t( 1.), Real_t(-1.) },
      { Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.),
        Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) },
      { Real_t(-1.), Real_t( 1.), Real_t(-1.), Real_t( 1.),
        Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) }
   } ;

   // Extract Views
   auto nodelist_v = domain.m_conn.m_nodelist ;
   auto x_v        = domain.m_nodes.m_x ;
   auto y_v        = domain.m_nodes.m_y ;
   auto z_v        = domain.m_nodes.m_z ;
   auto xd_v       = domain.m_nodes.m_xd ;
   auto yd_v       = domain.m_nodes.m_yd ;
   auto zd_v       = domain.m_nodes.m_zd ;
   auto v_v        = domain.m_elems.m_v ;
   auto volo_v     = domain.m_elems.m_volo ;
   auto ss_v       = domain.m_elems.m_ss ;
   auto elemMass_v = domain.m_elems.m_elemMass ;
   auto fx_v       = domain.m_nodes.m_fx ;
   auto fy_v       = domain.m_nodes.m_fy ;
   auto fz_v       = domain.m_nodes.m_fz ;

   if (hgcoef > Real_t(0.)) {
      /* P2-B: ScatterView — scatter hourglass forces directly to node forces.
         Replaces hg_fx/fy/fz scratch buffers + separate gather kernel.
         P2-A: LaunchBounds<256,2> → compiler targets ≤128 regs/thread (was 168)
               → SM occupancy 25% → 50%+; no-op on CPU backends. */
      auto fx_scatter = Kokkos::Experimental::create_scatter_view(fx_v) ;
      auto fy_scatter = Kokkos::Experimental::create_scatter_view(fy_v) ;
      auto fz_scatter = Kokkos::Experimental::create_scatter_view(fz_v) ;

      using hg_policy_t = Kokkos::RangePolicy<Kokkos::LaunchBounds<256, 2>>;
      Kokkos::parallel_for("CalcHourglassControlForElems", hg_policy_t(0, numElem),
                           KOKKOS_LAMBDA(Index_t i) {
         Real_t x1[8], y1[8], z1[8] ;
         Real_t pfx[8], pfy[8], pfz[8] ;

         const Index_t* elemToNode = nodelist_v.data() + 8*i ;

         // Inline CollectDomainNodesToElemNodes
         for (Index_t n = 0; n < 8; ++n) {
            Index_t gn = elemToNode[n] ;
            x1[n] = x_v(gn) ;
            y1[n] = y_v(gn) ;
            z1[n] = z_v(gn) ;
         }

         CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1) ;

         determ(i) = volo_v(i) * v_v(i) ;

         if (v_v(i) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive volume in CalcHourglassControlForElems") ;

         Real_t hgfx[8], hgfy[8], hgfz[8] ;
         Real_t hourgam[8][4] ;
         Real_t xd1[8], yd1[8], zd1[8] ;

         Real_t volinv = Real_t(1.0) / determ(i) ;

         for (Index_t i1 = 0; i1 < 4; ++i1) {
            Real_t hourmodx =
               x1[0]*gamma[i1][0] + x1[1]*gamma[i1][1] +
               x1[2]*gamma[i1][2] + x1[3]*gamma[i1][3] +
               x1[4]*gamma[i1][4] + x1[5]*gamma[i1][5] +
               x1[6]*gamma[i1][6] + x1[7]*gamma[i1][7] ;
            Real_t hourmody =
               y1[0]*gamma[i1][0] + y1[1]*gamma[i1][1] +
               y1[2]*gamma[i1][2] + y1[3]*gamma[i1][3] +
               y1[4]*gamma[i1][4] + y1[5]*gamma[i1][5] +
               y1[6]*gamma[i1][6] + y1[7]*gamma[i1][7] ;
            Real_t hourmodz =
               z1[0]*gamma[i1][0] + z1[1]*gamma[i1][1] +
               z1[2]*gamma[i1][2] + z1[3]*gamma[i1][3] +
               z1[4]*gamma[i1][4] + z1[5]*gamma[i1][5] +
               z1[6]*gamma[i1][6] + z1[7]*gamma[i1][7] ;

            hourgam[0][i1] = gamma[i1][0] - volinv*(pfx[0]*hourmodx + pfy[0]*hourmody + pfz[0]*hourmodz);
            hourgam[1][i1] = gamma[i1][1] - volinv*(pfx[1]*hourmodx + pfy[1]*hourmody + pfz[1]*hourmodz);
            hourgam[2][i1] = gamma[i1][2] - volinv*(pfx[2]*hourmodx + pfy[2]*hourmody + pfz[2]*hourmodz);
            hourgam[3][i1] = gamma[i1][3] - volinv*(pfx[3]*hourmodx + pfy[3]*hourmody + pfz[3]*hourmodz);
            hourgam[4][i1] = gamma[i1][4] - volinv*(pfx[4]*hourmodx + pfy[4]*hourmody + pfz[4]*hourmodz);
            hourgam[5][i1] = gamma[i1][5] - volinv*(pfx[5]*hourmodx + pfy[5]*hourmody + pfz[5]*hourmodz);
            hourgam[6][i1] = gamma[i1][6] - volinv*(pfx[6]*hourmodx + pfy[6]*hourmody + pfz[6]*hourmodz);
            hourgam[7][i1] = gamma[i1][7] - volinv*(pfx[7]*hourmodx + pfy[7]*hourmody + pfz[7]*hourmodz);
         }

         Real_t ss1      = ss_v(i) ;
         Real_t mass1    = elemMass_v(i) ;
         Real_t volume13 = Kokkos::cbrt(determ(i)) ;

         for (Index_t n = 0; n < 8; ++n) {
            Index_t gn = elemToNode[n] ;
            xd1[n] = xd_v(gn) ;
            yd1[n] = yd_v(gn) ;
            zd1[n] = zd_v(gn) ;
         }

         Real_t coefficient = -hgcoef * Real_t(0.01) * ss1 * mass1 / volume13 ;

         CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient, hgfx, hgfy, hgfz) ;

         /* P2-B: scatter directly to node forces via atomicAdd (CUDA) */
         auto fx_acc = fx_scatter.access() ;
         auto fy_acc = fy_scatter.access() ;
         auto fz_acc = fz_scatter.access() ;
         for (Index_t n = 0; n < 8; ++n) {
            fx_acc(elemToNode[n]) += hgfx[n] ;
            fy_acc(elemToNode[n]) += hgfy[n] ;
            fz_acc(elemToNode[n]) += hgfz[n] ;
         }
      });

      Kokkos::Experimental::contribute(fx_v, fx_scatter) ;
      Kokkos::Experimental::contribute(fy_v, fy_scatter) ;
      Kokkos::Experimental::contribute(fz_v, fz_scatter) ;

   } else {
      // hgcoef == 0: still need to fill determ (used by volume check)
      Kokkos::parallel_for("CalcHourglassControlForElems_determ", numElem,
                           KOKKOS_LAMBDA(Index_t i) {
         determ(i) = volo_v(i) * v_v(i) ;
         if (v_v(i) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive volume in CalcHourglassControlForElems") ;
      });
   }
}

/******************************************/

void CalcVolumeForceForElems(Domain& domain)
{
   Index_t numElem = domain.numElem() ;
   if (numElem != 0) {
      Real_t hgcoef = domain.hgcoef() ;

      // determ: pre-allocated scratch View (P1 — no per-step cudaMalloc).
      auto& determ = domain.m_elems.m_scratch.m_determ ;

      IntegrateStressForElems(domain, determ, numElem) ;

      // Check for negative element volume
      Kokkos::parallel_for("CalcVolumeForceForElems_check", numElem,
                           KOKKOS_LAMBDA(Index_t k) {
         if (determ(k) <= Real_t(0.0))
            Kokkos::abort("VolumeError: non-positive Jacobian determinant") ;
      });

      CalcHourglassControlForElems(domain, determ, hgcoef) ;
   }
}
